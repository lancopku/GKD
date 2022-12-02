from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from loss import distillation_loss, patience_loss, diff_loss
from .model_parts.embedding import part_embedding
from .model_parts.encoder import part_encoder
from .model_parts.classifier import part_classifier
from diff_signal import logits_to_signal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from copy import deepcopy


normalized_diff = True


class KDBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, teacher, T, strategy, num_layers, alpha, beta,**extra):
        self.config = deepcopy(teacher.config)
        self.config.update({"num_hidden_layers": num_layers})
        self.config.update({"attention_probs_dropout_prob":0,"hidden_dropout_prob":0})
        super(KDBertForSequenceClassification, self).__init__(self.config)

        self.alpha = alpha
        self.T = T
        self.beta = beta
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

        teacher_diff_signal = logits_to_signal(teacher_result.logits)
        teacher_emb_grad = torch.autograd.grad(teacher_diff_signal, teacher_input_emb, only_inputs=True)[0].detach().clone()

        teacher_scores = teacher_result.logits.detach().clone()
        self.teacher.train()

        student_encoder_inputs, labels = part_embedding(self, **inputs)
        student_input_emb = student_encoder_inputs["hidden_states"]
        student_input_emb.requires_grad = True
        student_result = part_encoder(self, **student_encoder_inputs)
        student_result = part_classifier(self, student_result, labels, student_encoder_inputs["return_dict"])

        student_diff_signal = logits_to_signal(student_result.logits)
        student_emb_grad = torch.autograd.grad(student_diff_signal, student_input_emb, only_inputs=True, create_graph=True)[0]


        student_scores = student_result.logits

        diff_kd_loss = diff_loss(teacher_diff=teacher_emb_grad, student_diff=student_emb_grad)

        vanilla_kd_loss, soft_loss, ce_loss = distillation_loss(student_scores, inputs["labels"], teacher_scores,
                                                                self.T, self.alpha)

        loss = vanilla_kd_loss + self.beta * diff_kd_loss
        student_result.loss = loss

        return student_result
