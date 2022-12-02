import argparse
from utils.prepare_teacher import get_teacher, get_tokenizer
from utils.others import set_seed
import json
import os
from utils.load_metric import load_metric
from transformers import TrainingArguments, Trainer
from utils.others import json_dump
from transformers import AutoTokenizer
from datasets import load_dataset
import time

# get args

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--gamma', type=float, default=0)
parser.add_argument('--task', type=str, default="mnli")
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--T', type=float, default=5)
parser.add_argument('--bs', type=int,
                    default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=4)
parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--strategy', type=str, default="gkd-cls")
parse_args = parser.parse_args()

# settings according to args
set_seed(parse_args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(parse_args.gpu)


glue_or_other_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
glue_tasks = glue_or_other_task_to_keys.keys()


# some preparation


def load_glue_data(args, tokenizer):
    task_name = args.task
    raw_datasets = load_dataset("glue", task_name)

    class data_args:
        def __init__(self, task_name):
            self.task_name = task_name
            self.root_folder = "./teacher"
            self.max_length = 128
            self.padding = "max_length"
            self.overwrite_cache = False

    _data_args = data_args(task_name)

    def tokenizer_function(sample, tokenizer, datasets, args=_data_args):
        task_name = args.task_name
        sentence1_key, sentence2_key = glue_or_other_task_to_keys[task_name]
        data = (
            (sample[sentence1_key],) if sentence2_key is None else (sample[sentence1_key], sample[sentence2_key])
        )
        padding = args.padding
        result = tokenizer(*data, padding=padding, max_length=args.max_length, truncation=True)
        return result

    preprocess_function = lambda x: tokenizer_function(x, tokenizer, raw_datasets)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation_matched" if task_name == "mnli" else "validation"]

    test_datasets = {task_name: tokenized_datasets["test"]} if task_name != "mnli" else {
        "mnli-m": tokenized_datasets["test_matched"],
        "mnli-mm": tokenized_datasets["test_mismatched"]}
    return {"train": train_dataset, "eval": eval_dataset, "test": test_datasets}


def load_data(args, tokenizer):
    task_name = args.task
    if task_name in glue_tasks:
        return load_glue_data(args, tokenizer)
    else:
        raise Exception("not implemented")


tokenizer = get_tokenizer(parse_args.task)
datasets = load_data(parse_args, tokenizer)

# prepare train
train_dataset = datasets["train"]
eval_dataset = datasets["eval"]

compute_metrics = load_metric(task_name=parse_args.task)

if parse_args.strategy == "kd":
    from model_architectures.vanilla_kd import KDBertForSequenceClassification
if parse_args.strategy == "pkd":
    from model_architectures.pkd import KDBertForSequenceClassification
if parse_args.strategy == "gkd":
    from model_architectures.gkd import KDBertForSequenceClassification
if parse_args.strategy == "gkd-cls":
    from model_architectures.multi_cls import KDBertForSequenceClassification
if parse_args.strategy == "gkd-cls-dropout":
    from model_architectures.multi_cls_dropout import KDBertForSequenceClassification

class args:
    def __init__(self, task_name):
        self.bs = parse_args.bs
        self.max_length = 128
        self.root_folder = "./teacher"
        self.alpha = parse_args.alpha
        self.beta = parse_args.beta
        self.gamma = parse_args.gamma
        self.T = parse_args.T
        self.num_layers = parse_args.layer
        self.num_epochs = parse_args.epoch
        self.padding = "max_length"
        self.task_name = task_name
        self.lr = parse_args.lr
        self.save_folder = "./student"
        self.save_args_path = "args.json"
        self.strategy = parse_args.strategy
        self.folder_path = "%s_%.1f_%.1f-%.1f_%.1f_%f_%d_seed_%d_layer_%d_%d" % (self.strategy,
                                                                                   self.alpha, self.T,
                                                                                   self.beta,
                                                                                   self.gamma,
                                                                                   self.lr,
                                                                                   parse_args.bs,
                                                                                   parse_args.seed,
                                                                                   self.num_layers,
                                                                                   int(time.time()))


# prepare for kd
train_args = args(parse_args.task)
teacher = get_teacher(train_args.task_name)
student_save_path = os.path.join(train_args.save_folder, train_args.task_name, train_args.folder_path)


student = KDBertForSequenceClassification(config=None, teacher=teacher, **train_args.__dict__)

# save args
os.makedirs(student_save_path, exist_ok=True)
with open(os.path.join(student_save_path, train_args.save_args_path), "w", encoding="utf-8") as f:
    json.dump(parse_args.__dict__, f, indent=4)

training_args = TrainingArguments(evaluation_strategy="epoch",
                                  num_train_epochs=train_args.num_epochs,
                                  output_dir=student_save_path,
                                  learning_rate=train_args.lr,
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  metric_for_best_model="accuracy",
                                  per_device_train_batch_size=train_args.bs,
                                  save_total_limit=1,
                                  seed=parse_args.seed)

trainer = Trainer(
    model=student,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# train and save
trainer.train()
trainer.model.teacher = None
trainer.save_model()
trainer.save_state()


# eval and save
eval_result = trainer.predict(eval_dataset)

with open(os.path.join(student_save_path, "eval_result.json"), "w") as f:
    json_dump(eval_result, f)
