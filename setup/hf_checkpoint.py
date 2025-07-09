CONSOLIDATED_CKPT_PATH = "/data/users/yangzho6/lingua/checkpoints/Llama3.23B/consolidated.00.pth" 
DCP_DIR_PATH = "/data/users/yangzho6/lingua/checkpoints/llama-3.2-3b-dcp" 

from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
torch_save_to_dcp(CONSOLIDATED_CKPT_PATH, DCP_DIR_PATH) 
