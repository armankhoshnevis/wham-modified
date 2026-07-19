set -o pipefail

mkdir -p runs/mac_smoke/logs

time python scripts/exp/train.py \
  --args.load conf/generated/mac_smoke_domain/c2f.yml \
  2>&1 | tee runs/mac_smoke/logs/domain_c2f.log