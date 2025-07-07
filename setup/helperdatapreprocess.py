from datasets import load_dataset 
import os 

def generate_textfield(example): 
    inputtext = "Question: {}\nAnswer: {}".format(example.pop("problem"), example.pop("generations")[0]) 
    return {"text": inputtext} 

def preprocess_data(datasetpath, outputpath): 
    dataset = load_dataset(datasetpath, "all", split = "train") 
    dataset = dataset.map(generate_textfield) 
    
    os.makedirs(outputpath, exist_ok=True) 
    dataset.to_parquet(outputpath) 

if __name__ == "__main__": 
    datasetpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/openr1_220k" 
    outputpath = "/fsx-storygen/jwzhao/yangzho6/lingua/setup/data/openr1_220k_processed.parquet" 
    preprocess_data(datasetpath, outputpath) 
