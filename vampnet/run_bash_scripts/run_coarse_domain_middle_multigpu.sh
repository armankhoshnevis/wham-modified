#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/middle_multigpu_domain/logs

time env \
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  PYTHONUNBUFFERED=1 \
  NCCL_DEBUG=WARN \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    scripts/exp/train.py \
    --args.load conf/generated/middle_multigpu_domain/coarse.yml \
  2>&1 | tee runs/middle_multigpu_domain/logs/coarse.log