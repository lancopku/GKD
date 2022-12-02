# README

## Installation
Run command below to install the environment
```bash
pip install -r requirements.txt
```


## Training
#### Teacher preparation

Before training the student, you need to first prepare four task-specific teachers. You can use the *run_glue.sh* to run *run_glue.py*（coppied from transformers package） to get the teacher.  

Or you can put you model at ./teacher folder. In this case, make sure it can be load by:

```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("./teacher/{task}")
```

Here {task} is in {"qnli","sst2","mnli","qqp"}. 



#### KD training

To train a student with kd methods, please use commands like below:



```
python run_kd.py --strategy {strategy} --task {task} --alpha {alpha} --T {T} --beta {beta} --gamma {gamma} --seed {seed}
```

| method                 | strategy          | necessary hyper-parameters |
| ---------------------- | ----------------- | -------------------------- |
| Vanilla KD             | "kd"              | alpha, T                   |
| BERT-PKD               | "pkd"             | alpha, T, beta             |
| GKD                    | "gkd"             | alpha, T, beta             |
| GKD-CLS                | "gkd-cls"         | alpha, T, beta, gamma      |
| GKD-CLS (with dropout) | "gkd-cls-dropout" | alpha, T, beta, gamma      |



