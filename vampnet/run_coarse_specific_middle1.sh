set -o pipefail

mkdir -p runs/middle1_test/logs

time python scripts/exp/train.py \
  --args.load conf/generated/middle1_specific/coarse.yml \
  2>&1 | tee runs/middle1_test/logs/specific_coarse.log