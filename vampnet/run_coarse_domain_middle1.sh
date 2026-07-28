set -o pipefail

mkdir -p runs/middle1_test/logs

time python scripts/exp/train.py \
  --args.load conf/generated/middle1_domain/coarse.yml \
  2>&1 | tee runs/middle1_test/logs/domain_coarse.log