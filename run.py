
import os
import gc
import shutil
import torch
import traceback
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

from dataloader import load_data
from constants import (
    HUGGINGFACE_ACCESS_TOKEN, 
    VOLUME_CACHE_DIR, 
    VOLUME_MODEL_CHECKPOINTS_BASE_PATH, 
    VOLUME_SAVED_MODELS_BASE_PATH, 
    VOLUME_METRICS_DIR
)

DEFAULT_MODEL_NAME = "mental/mental-bert-base-uncased"
DEFAULT_DATA_NAME = "huggingface_reddit"
DEFAULT_LABEL_COUNT = 5
BATCH_SIZE = 16
CLEAR_CACHE = False

def clear_cuda_cache(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def get_model_metric_path (model_name, dataset_name):
    return f"{VOLUME_METRICS_DIR}/{model_name.replace('/', '_')}_{dataset_name}_metrics.txt"

def compute_metrics(model_name, dataset_name):
    def inner_function(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric1, metric2, metric3, metric4 = evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1"), evaluate.load("accuracy")
        
        precision = metric1.compute(predictions=predictions, references=labels,average='weighted')
        recall = metric2.compute(predictions=predictions, references=labels,average='weighted')
        f1 = metric3.compute(predictions=predictions, references=labels,average='weighted')
        accuracy = metric4.compute(predictions=predictions, references=labels)

        with open(get_model_metric_path(model_name=model_name, dataset_name=dataset_name), "a+") as f:
            f.write(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}\n")
        
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
    return inner_function


def train(**inputs):
    gc.collect()
    torch.cuda.empty_cache()
    
    if CLEAR_CACHE:
        if os.path.isdir(VOLUME_CACHE_DIR):
            shutil.rmtree(VOLUME_CACHE_DIR)
            
    if not os.path.exists(VOLUME_CACHE_DIR):
        os.mkdir(VOLUME_CACHE_DIR)

    
    model_name = inputs.get("model_name", DEFAULT_MODEL_NAME)
    dataset = inputs.get("dataset", DEFAULT_DATA_NAME)
    label_count = int(inputs.get("label_count", DEFAULT_LABEL_COUNT))
    batch_size = int(inputs.get("batch_size", BATCH_SIZE))

    if os.path.exists(get_model_metric_path(model_name=model_name, dataset_name=dataset)):
        os.remove(get_model_metric_path(model_name=model_name, dataset_name=dataset))
    
    print(f"Model name: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Label count: {label_count}")
    print(f"Batch size: {batch_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=label_count,
        use_auth_token=HUGGINGFACE_ACCESS_TOKEN,
        # cache_dir=VOLUME_CACHE_DIR
    ).to(device)
    
    tokenized_datasets_train, tokenized_datasets_test = load_data(dataset, model_name)
    
    try:
        print("Beginning training...")
        if not os.path.isdir(VOLUME_MODEL_CHECKPOINTS_BASE_PATH):
            os.mkdir(VOLUME_MODEL_CHECKPOINTS_BASE_PATH)
        if not os.path.isdir(VOLUME_SAVED_MODELS_BASE_PATH):
            os.mkdir(VOLUME_SAVED_MODELS_BASE_PATH)
        
        default_args = {
            "output_dir": VOLUME_MODEL_CHECKPOINTS_BASE_PATH,
            "evaluation_strategy": "epoch",
            "num_train_epochs": 3,
            "log_level": "error",
            "report_to": "none",
        }
        
        training_args = TrainingArguments(per_device_train_batch_size=batch_size, **default_args)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets_train,
            eval_dataset=tokenized_datasets_test,
            compute_metrics=compute_metrics(model_name, dataset),
        )

        trainer.train()
        trainer.save_model(f"{VOLUME_SAVED_MODELS_BASE_PATH}/{dataset}-{model_name.replace('/','-')}")
        
    except:
        print(traceback.format_exc())
        clear_cuda_cache(model)
        
        print("CLOSING APP")


if __name__ == "__main__":    
    train()