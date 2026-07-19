set -o pipefail

mkdir -p runs/mac_smoke/logs

time python scripts/exp/train.py \
  --args.load conf/generated/mac_smoke_species/coarse.yml \
  2>&1 | tee runs/mac_smoke/logs/species_coarse.log