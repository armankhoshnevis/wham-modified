set -o pipefail

mkdir -p runs/smoke_test/logs

time python scripts/exp/train.py \
  --args.load conf/generated/smoke_specific/coarse.yml \
  2>&1 | tee runs/smoke_test/logs/specific_coarse.log