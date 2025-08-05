from datasets import load_dataset 
from transformers import AutoTokenizer 
import os 

from huggingface_hub import snapshot_download 
import requests 
import time 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct") 

def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,  # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}") 

def generate_textfield(example): 
    inputtext = "Question: {}\nAnswer: {}".format(example.pop("question"), example.pop("answer")) 
    return {"text": inputtext} 

def generate_textfieldtwo(example): 
    inputtext = tokenizer.apply_chat_template(example["messages"], tokenize=False) 
    return {"text": inputtext} 

def preprocess_data(datasetpath, outputpath, name): 
    if name == "a-m_team": 
        dataset = load_dataset(datasetpath, split = "train") 
        dataset = dataset.map(generate_textfield) 
    elif "chat" in name: 
        dataset = load_dataset(datasetpath, split = "train") 
        dataset = dataset.map(generate_textfieldtwo) 
    else: 
        raise ValueError("Invalid dataset name") 
    
    # os.makedirs(outputpath, exist_ok=True) 
    # dataset.to_parquet(outputpath) 
    dataset.to_json(outputpath) 

if __name__ == "__main__": 
    download_dataset()
    datasetpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/a-m_team_primitive" 
    outputpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/a-m_team/a-m_team_primitive.jsonl" 
    preprocess_data(datasetpath, outputpath) 
