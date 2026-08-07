#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/smoke_multigpu_specific/logs

time env \
  CUDA_VISIBLE_DEVICES=1,2,3,4 \
  PYTHONUNBUFFERED=1 \
  NCCL_DEBUG=INFO \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    scripts/exp/train.py \
    --args.load conf/generated/smoke_multigpu_specific/c2f.yml \
  2>&1 | tee runs/smoke_multigpu_specific/logs/c2f.log