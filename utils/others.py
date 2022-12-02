import json
import numpy as np
import transformers
import os
import torch
import random


def json_dump(obj, f):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                obj[key] = value.tolist()
    elif isinstance(obj, transformers.trainer_utils.PredictionOutput):
        obj = dict(obj._asdict())
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                obj[key] = value.tolist()
    json.dump(obj, f, indent=4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


