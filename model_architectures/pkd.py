from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
import torch
from loss import distillation_loss, patience_loss

from copy import copy



def patience_filter(patience, obj="teacher", num_layers=6):
    if num_layers == 6:
        layer_ids = [2, 4, 6, 8, 10] if obj == "teacher" else [1, 2, 3, 4, 5]
    else:
        raise NotImplementedError
    patience = [patience[_][:, 0:1, :] for _ in layer_ids]
    patience = torch.cat(patience, dim=1)
    return patience


class KDBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self,teacher, T, strategy, num_layers, alpha=0., beta=0., **extra):
        self.config = copy(teacher.config)
        self.config.update({"num_hidden_layers": num_layers})
        super(KDBertForSequenceClassification, self).__init__(self.config)
        self.alpha = alpha
        self.T = T
        self.beta = beta
        self.strategy = strategy
        self.num_layers = num_layers
        self.teacher = teacher

        self._part_init_weights()
        self.freeze_teacher()

    def freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad=False

    def _part_init_weights(self):
        self.bert.embeddings.load_state_dict(self.teacher.bert.embeddings.state_dict())

        for i in range(self.num_layers):
            self.bert.encoder.layer[i].load_state_dict(self.teacher.bert.encoder.layer[i].state_dict())

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

        self.teacher.eval()
        inputs["output_hidden_states"] = True
        student_result = super(KDBertForSequenceClassification, self).forward(**inputs)
        with torch.no_grad():
            teacher_result = self.teacher(**inputs)
        self.teacher.train()

        student_scores = student_result.logits
        teacher_scores = teacher_result.logits.detach().clone()

        student_patience = student_result.hidden_states
        student_patience = patience_filter(student_patience, "student", self.num_layers)
        teacher_patience = teacher_result.hidden_states
        teacher_patience = patience_filter(teacher_patience, "teacher", self.num_layers).detach().clone()

        vanilla_kd_loss, soft_loss, ce_loss = distillation_loss(student_scores, inputs["labels"], teacher_scores, self.T, self.alpha)
        if self.beta>0:
            patient_kd_loss = patience_loss(teacher_patience, student_patience, normalized_patience=True)
        else:
            patient_kd_loss = 0.0
        loss = vanilla_kd_loss + self.beta * patient_kd_loss
        student_result.loss = loss

        return student_result
