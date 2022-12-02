import numpy as np
import datasets
from transformers import EvalPrediction

glue_tasks = {"qqp", "qnli", "sst2", "mnli"}

def load_metric(task_name):
    if task_name in glue_tasks:
        metric = datasets.load_metric("glue", task_name)
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        return compute_metrics

    else:
        raise Exception("not implemented")