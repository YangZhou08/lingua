MANIFUSE_BUCKET=xr_core_ai_asl_llm
# MANIFUSE_SRC_PATH=tree/pretrain_yangzho6/
# MANIFUSE_FUSE_DST=/data/users/yangzho6/lingua/mount_pretrain_yangzho6/
STORAGE_FBPKG_OVERRIDE=oil.oilfs:stable

# source mount.sh

# Download OILFS fbpkg (most environments will already have this)
# fbpkg fetch oil.oilfs:stable

# Create an empty directory for the mount (e.g. /mnt/manifold)
# mkdir -p /home/yangzho6/pretrain_yangzho6

# Execute the OILFS wrapper for a bucket
# Can supply an optional path prefix into the bucket (e.g. 'tree/')
# ./oilfs-wrapper --bin ./oilfs --profile=manifold manifold://xr_core_ai_asl_llm/tree/pretrain_yangzho6 /home/yangzho6/pretrain_yangzho6

# torchx run \
#   --scheduler_args="fbpkg_ids=torchx_conda_mount:stable,oil.oilfs:stable"  \
#   fb.conda.torchrun \
#   --env "DISABLE_NFS=1,DISABLE_OILFS=1,LD_PRELOAD=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so,STORAGE_FBPKG_OVERRIDE=oil.oilfs:stable,MANIFUSE_BUCKET=$MANIFUSE_BUCKET,MANIFUSE_SRC_PATH=$MANIFUSE_SRC_PATH,MANIFUSE_DST_PATH=$MANIFUSE_DST_PATH" \
#   --h grandteton \
#   --run_as_root True \
#   -- \
#   --no-python --nnodes=1 --nproc-per-node=8 \
#   ./run.sh readingtrial.py

# mkdir -p $MANIFUSE_FUSE_DST

torchx run \
  --scheduler_args=tags=ads_ranking_taxonomy_ads_ai_am,bypass_cpu_job_tc_validator,fbpkg_ids=torchx_conda_mount:stable,oil.oilfs:stable, \
  fb.conda.torchrun \
  --env "DISABLE_NFS=1,DISABLE_OILFS=1,STORAGE_FBPKG_OVERRIDE=oil.oilfs:stable,MANIFUSE_BUCKET=$MANIFUSE_BUCKET" \
  --h t1 \
  --run_as_root True \
  -- \
  --no-python --nnodes=1 --nproc-per-node=1 \
  ./run.sh readingtrial.py
