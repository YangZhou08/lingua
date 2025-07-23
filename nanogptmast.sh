JOB_DATA_PROJECT=monetization_genai_platform
JOB_ONCALL=monetization_genai_platform

# torchx run \
#   --scheduler=mast_conda \
#   --scheduler_args="hpcIdentity=${JOB_DATA_PROJECT},hpcJobOncall=${JOB_ONCALL},rmAttribution = ads_global_tc_ai_creatives,modelTypeName= rl_xr_research_llm_reasoning,workspace_fbpkg_name=metaconda_demo,fbpkg_ids=conda_mast_core:stable,forceSingleRegion=False,tags=ads_ranking_taxonomy_ads_ai_am,bypass_cpu_job_tc_validator"  \
#   fb.conda.torchrun \
#   --h t1 \
#   --run_as_root True \
#   -- \
#   --no-python --nnodes=2 --nproc-per-node=1 ./run.sh

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
