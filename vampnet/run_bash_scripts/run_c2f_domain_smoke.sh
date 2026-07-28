set -o pipefail

mkdir -p runs/smoke_test/logs

time python scripts/exp/train.py \
  --args.load conf/generated/smoke_domain/c2f.yml \
  2>&1 | tee runs/smoke_test/logs/domain_c2f.log