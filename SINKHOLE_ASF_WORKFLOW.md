# Sinkhole ASF Workflow

This repository now includes:

- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py`
- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/train_sinkhole_precursor_baseline.py`
- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/config/event_split.yaml`
- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/scripts/roar_night1_derived.sh`
- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/scripts/roar_night2_slc_subset.sh`

It discovers Sentinel-1 (priority), ARIA GUNW, and OPERA S1 displacement products for 8 Pennsylvania sinkhole events using a default 12-month pre-event and 3-month post-event search window.

## Install

```bash
python3 -m pip install asf_search pandas requests
```

## Run Discovery (metadata/manifests only)

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py
```

## Run Discovery + Browse Quicklooks

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py --download-browse 15
```

## Run Discovery + Product Downloads (interactive auth)

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py --download-products --auth-mode auto
```

When `--download-products` is enabled, the script:
- checks env vars first (`EARTHDATA_TOKEN`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `ASF_*` equivalents),
- if not found, prompts you for token or username/password.

## Outputs

Default output directory:

- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/sinkhole_event_discovery`

Main files:

- `events_used.csv`: event catalog used in run
- `manifest_summary.csv`: counts by event and dataset
- `all_manifests.csv`: combined product metadata
- `run_metadata.json`: run configuration
- `downloads_log.csv`: only when `--download-products` is used
- `qc/coverage_report.csv`: per event x dataset coverage checks
- `qc/missingness_report.csv`: null/duplicate stats
- `qc/parse_errors.csv`: categorized parse/download issues
- `qc/run_summary.json`: acceptance criteria summary

Per-event directories include dataset manifest CSVs and optional browse/download folders.

## Staged Approach (recommended)

1. Build manifests for all events (no bulk product download):

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py --skip-counts
```

2. Pick one event and one dataset, cap file count:

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py \
  --skip-counts \
  --download-products \
  --auth-mode auto \
  --event-id 2023-08-16_eisenhower_parking_deck_penn_state \
  --dataset SENTINEL1_SLC \
  --max-downloads-per-manifest 5
```

3. Expand in batches once storage/runtime looks good.

Useful flags for staging:
- `--event-id <id>` (repeat to include more events)
- `--dataset SENTINEL1_SLC|ARIA_S1_GUNW|OPERA_S1_DISP` (repeat to include more datasets)
- `--max-downloads-per-manifest N`
- `--max-total-gb 50`
- `--prefer-derived` (skip SLC unless explicitly requested)
- `--qc-only` (generate QC outputs only from existing manifests)

## Locked Split

This split is defined in:

- `/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/config/event_split.yaml`

Resolved split files are written under:

- `outputs/sinkhole_event_discovery/qc/event_split_resolved.csv`
- `outputs/sinkhole_event_discovery/qc/event_split_resolved.json`

## Overnight Cadence

Night 1 (derived first + 50 GB cap + QC):

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py \
  --skip-counts \
  --download-products \
  --auth-mode auto \
  --prefer-derived \
  --max-total-gb 50
```

Night 2 (limited SLC for two training events, cap 5 each):

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py \
  --skip-counts \
  --download-products \
  --auth-mode auto \
  --dataset SENTINEL1_SLC \
  --event-id 2020-08-28_packer_twp_carbon_county \
  --event-id 2023-10-30_130_sickler_hill_rd_luzerne \
  --max-downloads-per-manifest 5 \
  --max-total-gb 50
```

QC-only rerun (no new downloads):

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/sinkhole_asf_discovery.py \
  --qc-only \
  --split-file /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/config/event_split.yaml
```

## Baseline ML Training

Train baseline precursor risk model (train split fit, val threshold tuning, final test report):

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/train_sinkhole_precursor_baseline.py \
  --discovery-outdir /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/sinkhole_event_discovery \
  --split-file /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/config/event_split.yaml \
  --outdir /Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/ml
```

Key ML outputs:

- `outputs/ml/feature_table.parquet` (normalized feature table)
- `outputs/ml/evaluation_summary.json` (precision/recall/F1, false alarms/year, lead-time, calibration bins)
- `outputs/ml/model_artifact.json`

## Local vs ROAR Cluster

- Use local laptop for metadata discovery and small test batches.
- Use ROAR for large SLC download and downstream InSAR processing, since storage and runtime demand can grow quickly.
