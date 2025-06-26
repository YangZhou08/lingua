import torch
import torch.distributed as dist

# 1. Initialize the distributed environment
dist.init_process_group(backend="nccl") # or "gloo", "mpi" 

# 2. Perform allreduce on the default process group (WORLD)
tensor = torch.tensor([1.0], device="cuda:{}".format(dist.get_rank())) 
dist.all_reduce(tensor) 
print(tensor) 
