set -o pipefail

time python scripts/exp/train.py \
  --args.load conf/generated/mac_smoke_species/coarse.yml \
  2>&1 | tee runs/mac_smoke/logs/species_coarse.log