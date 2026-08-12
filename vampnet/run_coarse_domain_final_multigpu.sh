#!/usr/bin/env bash
set -euo pipefail

config_path="conf/in_cabin_4gpu/domain/coarse.yml"
log_dir="runs/in_cabin_4gpu/domain/logs"

mkdir -p "${log_dir}"
time env \
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  PYTHONUNBUFFERED=1 \
  NCCL_DEBUG=WARN \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    scripts/exp/train.py \
    --args.load "${config_path}" \
  2>&1 | tee "${log_dir}/coarse.log"
