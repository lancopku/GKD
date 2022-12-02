import transformers
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions



def part_encoder(self: transformers.BertForSequenceClassification, **inputs):
    encoder_outputs = self.bert.encoder(**inputs)
    sequence_output = encoder_outputs[0]
    pooled_output = self.bert.pooler(sequence_output) if self.bert.pooler is not None else None

    return_dict = inputs["return_dict"]
    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )



