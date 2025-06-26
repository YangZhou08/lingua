CONSOLIDATED_CKPT_PATH = "/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/Llama-3.2-3B/original/consolidated.00.pth" 
DCP_DIR_PATH = "/fsx-storygen/jwzhao/yangzho6/lingua/checkpoints/debug/checkpoints/Llama-3.2-3Bdcp" 

from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
torch_save_to_dcp(CONSOLIDATED_CKPT_PATH, DCP_DIR_PATH) 
