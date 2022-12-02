from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig
import json
import os

def get_teacher(task):
    model_path = os.path.join("./teacher/%s"%task)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def get_tokenizer(task):
    with open("./settings.json","r",encoding="utf-8") as f:
        settings = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(settings["default_tokenizer_name"])
    return tokenizer

def get_config(task,layer=12):
    model_path = os.path.join("./teacher/%s" % task)
    config =  AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = layer
    return config


