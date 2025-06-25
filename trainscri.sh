# stool stands for SLURM tool !
# python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition> 
# if you want to launch locally you can use torchrun
torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml 
# or you can also launch on 1 GPU
# python -m apps.main.train config=apps/main/configs/debug.yaml 
