MANIFUSE_BUCKET=xr_core_ai_asl_llm
# MANIFUSE_SRC_PATH=tree/pretrain_yangzho6/
# MANIFUSE_FUSE_DST=/data/users/yangzho6/lingua/mount_pretrain_yangzho6/
STORAGE_FBPKG_OVERRIDE=oil.oilfs:stable

# torchx run \
#   --scheduler_args="fbpkg_ids=torchx_conda_mount:stable,oil.oilfs:stable"  \
#   fb.conda.torchrun \
#   --env "DISABLE_NFS=1;DISABLE_OILFS=1;MANIFUSE_BUCKET=demo_torchx_bucket,LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so,TRITON_LIBCUDA_PATH=/usr/local/fbcode/platform010/lib/" \
#   --h grandteton \
#   --run_as_root True \
#   -- \
#   --nnodes=1 --nproc-per-node=8 \
#   -m apps.main.train config=apps/main/xxconfigs/webonly.yaml

torchx run \
  --scheduler_args=tags=ads_ranking_taxonomy_ads_ai_am,bypass_cpu_job_tc_validator,fbpkg_ids=torchx_conda_mount:stable,oil.oilfs:stable, \
  fb.conda.torchrun \
  --env "DISABLE_NFS=1,DISABLE_OILFS=1,STORAGE_FBPKG_OVERRIDE=oil.oilfs:stable,MANIFUSE_BUCKET=$MANIFUSE_BUCKET,LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so,TRITON_LIBCUDA_PATH=/usr/local/fbcode/platform010/lib/" \
  --h grandteton \
  --run_as_root True \
  -- \
  --no-python --nnodes=1 --nproc-per-node=8 \
  ./run.sh -m apps.main.train config=apps/main/xxconfigs/webonly.yaml
