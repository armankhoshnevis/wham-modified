set -o pipefail

time python scripts/exp/train.py \
  --args.load conf/generated/mac_smoke_domain/coarse.yml \
  2>&1 | tee runs/mac_smoke/logs/domain_coarse.log