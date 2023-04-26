import pickle
import os
import zipfile

from transformers import AutoTokenizer
from constants import VOLUME_DATASETS_BASE_PATH, HUGGINGFACE_ACCESS_TOKEN, VOLUME_CACHE_DIR
from datasets import Dataset

def unzip_data(dataset_name: str):
    dir_name = f"{VOLUME_DATASETS_BASE_PATH}/{dataset_name}"
    
    if not os.path.isdir(dir_name):
        print("Unzipping data...")
        with zipfile.ZipFile(f"{dir_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(dir_name)
        
        return

    print(f"Data {dir_name} already unzipped...")


def tokenize_function(tokenizer):
    def inner_tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)
    return inner_tokenize_function


def load_data(dataset_name: str = "eRisk",  model_name: str = "nlpie/bio-mobilebert"):
    unzip_data(dataset_name)
    
    with open(f"{VOLUME_DATASETS_BASE_PATH}/{dataset_name}/{dataset_name}_y_train.pkl", 'rb') as f:
        data_train_y = pickle.load(f)

    with open(f"{VOLUME_DATASETS_BASE_PATH}/{dataset_name}/{dataset_name}_X_train.pkl", 'rb') as f:
        data_train_X = pickle.load(f)

    with open(f"{VOLUME_DATASETS_BASE_PATH}/{dataset_name}/{dataset_name}_y_test.pkl", 'rb') as f:
        data_test_y = pickle.load(f)

    with open(f"{VOLUME_DATASETS_BASE_PATH}/{dataset_name}/{dataset_name}_X_test.pkl", 'rb') as f:
        data_test_X = pickle.load(f)
    
    dataset_train = list()
    dataset_test = list()

    for text, label in zip(data_train_X, data_train_y):
        dataset_train.append({'label': int(label),'text': str(text)})
    
    for text, label in zip(data_test_X, data_test_y):
        dataset_test.append({'label': int(label),'text': str(text)})
    
    dataset_hf_train = Dataset.from_list(dataset_train)
    dataset_hf_test = Dataset.from_list(dataset_test)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=HUGGINGFACE_ACCESS_TOKEN,
        model_max_length=128,
        # cache_dir=VOLUME_CACHE_DIR
    )
    
    tokenized_datasets_train = dataset_hf_train.map(
        tokenize_function(tokenizer), 
        batched=True,
        # cache_file_name=f"{VOLUME_CACHE_DIR}/{dataset_name}_train_cache.cache"
    )
    tokenized_datasets_test = dataset_hf_test.map(
        tokenize_function(tokenizer), 
        batched=True, 
        # cache_file_name=f"{VOLUME_CACHE_DIR}/{dataset_name}_test_cache.cache"
    )

    return tokenized_datasets_train, tokenized_datasets_test

if __name__ == "__main__":
    dataset_hf_train, dataset_hf_test = load_data(
        "eRisk",
        "nlpie/tiny-clinicalbert"
    )
    print(dataset_hf_train)
    print(dataset_hf_test)
    