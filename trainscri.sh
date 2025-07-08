# stool stands for SLURM tool ! 
export LD_PRELOAD=/usr/local/cuda-12.<version>/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH 
huggingface-cli login --token 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 
wandb login --relogin fbb26fc8718b8e58d743b5cdcabaa2396656f773 

# python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition> 
# if you want to launch locally you can use torchrun 

torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/webonly.yaml 
# or you can also launch on 1 GPU
# python -m apps.main.train config=apps/main/configs/debug.yaml 
