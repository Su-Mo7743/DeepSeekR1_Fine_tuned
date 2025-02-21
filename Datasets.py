from datasets import load_dataset
from huggingface_hub import login

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

token = input("Please enter your HF API token: ")
login(token,add_to_git_credential=True)

def Dataset_load(url):
    ds = load_dataset(url,
                      cache_dir="E:\Ollma\Ollama\Datasets")
    if ds['train'][0]['question']:
        print("Dataset loaded")
    return ds




