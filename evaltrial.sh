# stool stands for SLURM tool ! 
export LD_PRELOAD=/usr/local/cuda-12.<version>/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH 
huggingface-cli login --token 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 

torchrun --nproc-per-node 8 -m apps.main.trialsimplet 
