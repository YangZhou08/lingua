#!/bin/bash

# Slurm batch script for job submission
#SBATCH --job-name=my_job          # Name of the job 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:8              # Request 8 GPUs
#SBATCH --cpus-per-task=12        # Request 12 CPUs per task
#SBATCH --partition=learn  # Specify the storygen_high queue/partition
#SBATCH -q storygen_high
#SBATCH --time=120:00:00           # Set time limit to 24 hours 
#SBATCH --output=job_%j.out       # Output file (%j is the job ID) 
#SBATCH --mem=512G 

# Optional: Load any required modules or set up environment
# module load cuda/11.8  # Example: Uncomment and adjust if needed

set -x

source /home/yangzho6/miniconda3/etc/profile.d/conda.sh
conda activate lingua_2506232 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Specify which GPUs to use
python -c "import torch; print(torch.cuda.is_available())"  # Check if CUDA is available 

huggingface-cli login --token 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 
wandb login --relogin fbb26fc8718b8e58d743b5cdcabaa2396656f773 

# Your job commands go here
# Replace this with the actual command you want to run
# Example: python my_script.py
# Add your specific commands here 

gpustat 
pwd 
cd /fsx-storygen/jwzhao/yangzho6/lingua 
git pull 

torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/webonly.yaml 
