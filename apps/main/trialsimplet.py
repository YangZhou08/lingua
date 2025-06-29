from lm_eval import simple_evaluate 
from termcolor import colored 

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer, 
) 

from apps.main.transformer import LMTransformer, LMTransformerArgs 
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints 
from apps.main.eval import EvalHarnessLM 
from dataclasses import asdict 
import json 

import yaml 

# consolidate_path = "/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/debug/checkpoints/Llama-3.2-3Bdcp" 
consolidate_path = "/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/debug/checkpoints/0000000100/consolidated" 
with open("/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/debug/evals/0000000100/config.yaml", "r") as file: 
    cfg = yaml.safe_load(file) 

print(colored("Loading model", "cyan")) 
model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
    consolidate_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
) 

print(colored("Model loaded", "cyan")) 
model.eval() 

generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer) 
wrap = EvalHarnessLM(generator) 

results = simple_evaluate(wrap, **asdict(cfg.harness)) 
print(results) 
with open("results.json", "w") as f: 
    f.write(json.dumps(results)) 
