from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import distillation_loss, patience_loss, diff_loss
from .model_parts.embedding import part_embedding
from .model_parts.encoder import part_encoder
from .model_parts.classifier import part_classifier
from diff_signal import logits_to_signal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from copy import deepcopy


normalized_diff = True

def patience_filter(patience, strategy="PKD-skip", obj="teacher", num_layers=6):
    if strategy == "PKD-skip" and num_layers == 6:
        layer_ids = [2, 4, 6, 8, 10] if obj == "teacher" else [1, 2, 3, 4, 5]
    else:
        raise NotImplementedError
    patience = [patience[_][:, 0:1, :] for _ in layer_ids]
    patience = torch.cat(patience, dim=1)
    return patience


class KDBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, teacher, T, strategy, num_layers, alpha, beta,gamma, freeze="y", **extra):
        self.config = deepcopy(teacher.config)
        self.config.update({"num_hidden_layers": num_layers})
        self.config.update({"attention_probs_dropout_prob": 0, "hidden_dropout_prob": 0})
        super(KDBertForSequenceClassification, self).__init__(self.config)
        self.alpha = alpha
        self.T = T
        self.beta = beta
        self.gamma = gamma
        self.strategy = strategy
        self.teacher = teacher
        self.num_layers = num_layers

        self._part_init_weights()

        self.freeze_teacher()
        self.freeze_embedding()

        if self.num_layers == 6:
            self.teacher_filter = [2, 4, 6, 8, 10]
            self.student_filter = [1, 2, 3, 4, 5]
        else:
            raise NotImplementedError

        self.embeddings = self.bert.embeddings

    def freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad = False

    def freeze_embedding(self):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False


    def _part_init_weights(self):
        self.bert.embeddings.dropout = nn.Dropout(0)
        self.teacher.bert.embeddings.dropout = nn.Dropout(0)
        self.bert.embeddings.load_state_dict(self.teacher.bert.embeddings.state_dict())
        for i in range(self.num_layers):
            self.bert.encoder.layer[i].load_state_dict(self.teacher.bert.encoder.layer[i].state_dict())


    def filter(self, tuple: List[torch.Tensor], idxs):
        return [tuple[i] for i in idxs]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                  "position_ids": position_ids,
                  "head_mask": head_mask, "inputs_embeds": inputs_embeds, "labels": labels,
                  "output_attentions": output_attentions,
                  "output_hidden_states": output_hidden_states, "return_dict": return_dict}

        if self.training == False:
            return super(KDBertForSequenceClassification, self).forward(**inputs)

        inputs["output_hidden_states"] = True

        self.teacher.eval()
        teacher_encoder_inputs, labels = part_embedding(self.teacher, **inputs)
        teacher_input_emb = teacher_encoder_inputs["hidden_states"]
        teacher_input_emb.requires_grad = True

        teacher_result = part_encoder(self.teacher, **teacher_encoder_inputs)
        teacher_result = part_classifier(self.teacher, teacher_result, labels,
                                         teacher_encoder_inputs["return_dict"])

        hidden_states = teacher_result.hidden_states
        hidden_states = self.filter(hidden_states, self.teacher_filter)
        t_states = [teacher_input_emb]
        t_states.extend(hidden_states)

        teacher_patience = teacher_result.hidden_states
        teacher_patience = patience_filter(teacher_patience, obj="teacher",
                                           num_layers=self.num_layers).detach().clone()

        teacher_diff_signal = logits_to_signal(teacher_result.logits)
        teacher_grads = torch.autograd.grad(teacher_diff_signal, t_states, only_inputs=True)
        teacher_grads = [teacher_grad.detach().clone() for teacher_grad in teacher_grads]
        teacher_scores = teacher_result.logits.detach().clone()
        self.teacher.train()

        student_encoder_inputs, labels = part_embedding(self, **inputs)
        student_input_emb = student_encoder_inputs["hidden_states"]

        student_input_emb.requires_grad = True
        student_result = part_encoder(self, **student_encoder_inputs)
        student_result = part_classifier(self, student_result, labels, student_encoder_inputs["return_dict"])
        student_patience = student_result.hidden_states
        student_patience = patience_filter(student_patience, obj="student",
                                           num_layers=self.num_layers)

        hidden_states = student_result.hidden_states
        hidden_states = self.filter(hidden_states, self.student_filter)
        s_states = [student_input_emb]
        s_states.extend(hidden_states)

        student_diff_signal = logits_to_signal(student_result.logits)

        student_grads = torch.autograd.grad(student_diff_signal, s_states, only_inputs=True, create_graph=True,
                                            retain_graph=True)

        student_scores = student_result.logits

        diff_kd_loss = 0.0


        for idx, (teacher_grad, student_grad) in enumerate(zip(teacher_grads, student_grads)):
            if idx == 0:
                _ = diff_loss(teacher_diff=teacher_grad, student_diff=student_grad)
            else:
                _ = diff_loss(teacher_diff=teacher_grad[:,0:1,:], student_diff=student_grad[:,0:1,:])
            diff_kd_loss =  diff_kd_loss + _

        vanilla_kd_loss, _, _ = distillation_loss(student_scores, inputs["labels"], teacher_scores, self.T, self.alpha,
                                                  )
        pkd_loss = patience_loss(student_patience=student_patience, teacher_patience=teacher_patience,
                                 normalized_patience=True)
        loss = vanilla_kd_loss + self.beta * pkd_loss + self.gamma * diff_kd_loss
        student_result.loss = loss

        return student_result
