import torch


def part_embedding(self, **inputs):
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    token_type_ids = inputs.get("token_type_ids", None)
    position_ids = inputs.get("position_ids", None)
    head_mask = inputs.get("head_mask", None)
    inputs_embeds = inputs.get("inputs_embeds", None)
    labels = inputs.get("labels", None)
    output_attentions = inputs.get("output_attentions", None)
    output_hidden_states = inputs.get("output_hidden_states", None)
    return_dict = inputs.get("return_dict", None)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict
    use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    past_key_values_length = 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(self.bert.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.bert.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

    embedding_output = self.bert.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )

    return {"hidden_states": embedding_output, "attention_mask": extended_attention_mask,
            "head_mask": head_mask,
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
            "past_key_values": None,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict},labels
