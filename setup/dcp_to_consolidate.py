CONSOLIDATE_FOLDER = "consolidated" 
CONSOLIDATE_NAME = "consolidated.00.pth" 
CONFIG_NAME = "params.json" 

from torch.distributed.checkpoint.format_utils import (
    torch_save_to_dcp,
    dcp_to_torch_save,
) 
from pathlib import Path 
from argparse import ArgumentParser 

parser = ArgumentParser() 
parser.add_argument("--ckpt_dir", type=str, required=True) 
args = parser.parse_args() 

def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        print(f"Consolidating to: {str(consolidate_path)}") 
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        print("Consolidated !") 
    return consolidate_path 

consolidate_path = consolidate_checkpoints(args.ckpt_dir) 
