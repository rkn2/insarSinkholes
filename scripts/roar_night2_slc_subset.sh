#!/usr/bin/env bash
set -euo pipefail

# Run on ROAR compute node via srun; requires EARTHDATA_TOKEN in env.
if [[ -z "${EARTHDATA_TOKEN:-}" ]]; then
  echo "EARTHDATA_TOKEN is not set. Export it before running."
  exit 1
fi

ROOT="/storage/home/rjn5308/sinkholes"
cd "$ROOT"

module load python/3.11.2
python3 -m pip install --user -q asf_search pandas requests pyyaml pyarrow

python3 sinkhole_asf_discovery.py \
  --skip-counts \
  --download-products \
  --auth-mode auto \
  --dataset SENTINEL1_SLC \
  --event-id 2020-08-28_packer_twp_carbon_county \
  --event-id 2023-10-30_130_sickler_hill_rd_luzerne \
  --max-downloads-per-manifest 5 \
  --max-total-gb 50 \
  --split-file config/event_split.yaml

python3 sinkhole_asf_discovery.py \
  --qc-only \
  --split-file config/event_split.yaml
