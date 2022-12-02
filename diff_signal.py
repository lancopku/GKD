import torch

def logits_to_signal(logits):
    probs = torch.softmax(logits,dim=-1)
    confidence = torch.max(probs,dim=-1)[0]
    return confidence.sum() # gradients of different samples can backprop simutaneously

