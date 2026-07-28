set -o pipefail

mkdir -p runs/middle1_test/logs

time python scripts/exp/train.py \
  --args.load conf/generated/middle1_specific/c2f.yml \
  2>&1 | tee runs/middle1_test/logs/specific_c2f.log