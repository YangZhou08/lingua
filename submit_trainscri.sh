#!/bin/bash

# Slurm batch script for job submission
#SBATCH --job-name=my_job          # Name of the job 
#SBATCH --nodes=2 
#SBATCH --gres=gpu:8              # Request 8 GPUs
#SBATCH --cpus-per-task=12        # Request 12 CPUs per task
#SBATCH --partition=learn  # Specify the storygen_high queue/partition 
#SBATCH -q storygen_high 
#SBATCH --account=storygen 
#SBATCH --time=1:00:00           # Set time limit to 24 hours 
#SBATCH --output=job_%j.out       # Output file (%j is the job ID) 
#SBATCH --mem=512G 

# Optional: Load any required modules or set up environment
# module load cuda/11.8  # Example: Uncomment and adjust if needed

set -x

source /home/yangzho6/miniconda3/etc/profile.d/conda.sh 
conda activate lingua_2506232 

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Specify which GPUs to use
# python -c "import torch; print(torch.cuda.is_available())"  # Check if CUDA is available 

# stool stands for SLURM tool ! 
export LD_PRELOAD=/usr/local/cuda-12.4/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH 
huggingface-cli login --token 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 
wandb login --relogin fbb26fc8718b8e58d743b5cdcabaa2396656f773 

# python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition> 
# if you want to launch locally you can use torchrun 

# torchrun --nnodes 2 --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml 
# or you can also launch on 1 GPU
# python -m apps.main.train config=apps/main/configs/debug.yaml 
# Launch one torchrun *per node*
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

srun --ntasks-per-node=1 bash -lc '
  export NODE_RANK=$SLURM_NODEID
  torchrun \
    --nnodes='"$SLURM_JOB_NUM_NODES"' \
    --nproc-per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m apps.main.train config=apps/main/configs/debug.yaml 
'
