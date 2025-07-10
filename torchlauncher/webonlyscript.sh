torchx run \
  --scheduler_args="fbpkg_ids=torchx_conda_mount:stable,oil.oilfs:stable"  \
  fb.conda.torchrun \
  --env "DISABLE_NFS=1;DISABLE_OILFS=1;MANIFUSE_BUCKET=demo_torchx_bucket,LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so" \
  --h grandteton \
  --run_as_root True \
  -- \
  --no-python --nnodes=1 --nproc-per-node=8 \
  ./run.sh train.py config/train_shakespeare_char.py \
  --gradient_accumulation_steps=8 \
  --tb_log=True \
  --out_dir='/mount/demo_torchx_bucket/out/${app_id}' 
