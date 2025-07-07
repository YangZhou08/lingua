from datasets import load_dataset 
import os 

def generate_textfield(example): 
    inputtext = "Question: {}\nAnswer: {}".format(example.pop("question"), example.pop("answer")) 
    return {"text": inputtext} 

def preprocess_data(datasetpath, outputpath): 
    dataset = load_dataset(datasetpath, "all", split = "train") 
    dataset = dataset.map(generate_textfield) 
    
    # os.makedirs(outputpath, exist_ok=True) 
    # dataset.to_parquet(outputpath) 
    dataset.to_json(outputpath) 

if __name__ == "__main__": 
    datasetpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/a-m_team_primitive" 
    outputpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/a-m_team/a-m_team_primitive.jsonl" 
    preprocess_data(datasetpath, outputpath) 
