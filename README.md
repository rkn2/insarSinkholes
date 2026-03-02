# InSAR Sinkhole Program (Multi-Event, ROAR-Ready)

This repository contains an end-to-end sinkhole monitoring workflow for Pennsylvania events:

1. ASF discovery + staged download (ARIA/OPERA/S1)
2. QC reporting + deterministic train/val/test split
3. Observed displacement extraction from ARIA `.nc`
4. Baseline model training and alert-policy calibration
5. De-leaked classifier training and trustworthiness audit

## Current Scope

- Event catalog: 8 sinkhole events (configured in script defaults + split file)
- Split contract:
  - `train`: 5 events
  - `val`: 1 event (Eisenhower)
  - `test`: 2 holdout events
- Primary operational target: low false alarms with lead-time-aware alerts

## Key Files

Core workflow scripts:

- `sinkhole_asf_discovery.py`
- `train_sinkhole_precursor_baseline.py`
- `calibrate_precursor_alert_policy.py`
- `train_precursor_classifier.py`
- `audit_model_trustworthiness.py`
- `extract_observed_displacement_from_aria.py`
- `extract_observed_displacement_from_aria_pairs.py`

Configs and run helpers:

- `config/event_split.yaml`
- `scripts/roar_night1_derived.sh`
- `scripts/roar_night2_slc_subset.sh`

## Environment

Recommended Python packages:

```bash
python3 -m pip install asf_search pandas numpy requests pyyaml pyarrow scikit-learn h5py
```

## Stage 1: Discovery + QC

### Derived-first discovery (no SLC by default)

```bash
python3 sinkhole_asf_discovery.py \
  --skip-counts \
  --prefer-derived \
  --max-results 1000 \
  --split-file config/event_split.yaml \
  --outdir outputs/sinkhole_event_discovery_derived_full
```

### QC-only rerun

```bash
python3 sinkhole_asf_discovery.py \
  --qc-only \
  --split-file config/event_split.yaml \
  --outdir outputs/sinkhole_event_discovery_derived_full
```

QC outputs:

- `qc/coverage_report.csv`
- `qc/missingness_report.csv`
- `qc/parse_errors.csv`
- `qc/run_summary.json`

## Stage 2: Observed Displacement Extraction (ARIA)

### Simple point extraction

```bash
python3 extract_observed_displacement_from_aria.py \
  --download-outdir /scratch/rjn5308/sinkholes/outputs/sinkhole_event_discovery \
  --manifest-root outputs/sinkhole_event_discovery_derived_full \
  --out-csv config/observed_displacement_aria.csv
```

### Pair-aware extraction (recommended)

```bash
python3 extract_observed_displacement_from_aria_pairs.py \
  --download-outdir /scratch/rjn5308/sinkholes/outputs/sinkhole_event_discovery \
  --download-outdir /scratch/rjn5308/sinkholes/outputs/sinkhole_event_discovery_aria_expand \
  --manifest-root outputs/sinkhole_event_discovery \
  --out-csv-raw config/observed_displacement_aria_pairs_raw.csv \
  --out-csv-agg config/observed_displacement_aria_pairs_agg.csv
```

## Stage 3: Baseline Modeling + Policy Calibration

```bash
python3 train_sinkhole_precursor_baseline.py \
  --discovery-outdir outputs/sinkhole_event_discovery_derived_full \
  --split-file config/event_split.yaml \
  --observed-displacement-csv config/observed_displacement_aria_pairs_agg.csv \
  --outdir outputs/ml/derived_full_observed_aria_pairs

python3 calibrate_precursor_alert_policy.py \
  --model-outdir outputs/ml/derived_full_observed_aria_pairs \
  --selection-mode event_cv \
  --max-false-alarms-per-year 3.0
```

## Stage 4: De-Leaked Classifier (Recommended Path)

Train de-leaked classifier by removing leakage-prone features and rebuilding labels from event window only:

```bash
python3 train_precursor_classifier.py \
  --feature-table outputs/ml/derived_full_observed_aria_pairs_relaxed_w8/feature_table.parquet \
  --outdir outputs/ml/classifier_v2_deleaked \
  --model hgb \
  --positive-class-weight 8 \
  --val-far-cap 3.0 \
  --relabel-window-days 60 \
  --exclude-features days_to_event robust_vel_z robust_accel_z

python3 calibrate_precursor_alert_policy.py \
  --model-outdir outputs/ml/classifier_v2_deleaked \
  --selection-mode event_cv \
  --max-false-alarms-per-year 3.0
```

Conservative deployed policy artifact:

- `outputs/ml/classifier_v2_deleaked/deployed_alert_policy.json`

## Stage 5: Trustworthiness Audit (Required)

```bash
python3 audit_model_trustworthiness.py \
  --model-outdir outputs/ml/classifier_v1_hgb
```

Audit outputs:

- `trustworthiness_audit.json`
- `trustworthiness_audit.md`

## ROAR Notes

- Run heavy downloads/extraction on compute nodes (`srun -p interactive`), not submit node.
- Use `/scratch` for large downloads.
- Keep Earthdata token in environment, not hardcoded in scripts.

## Reproducibility Artifacts

- `outputs/roar_minimal/REPRODUCIBILITY_TRAIL.txt`
- `outputs/roar_minimal/analysis_ready_index.csv`
- `outputs/roar_minimal/analysis_ready_index_summary.json`

These include command history, run IDs, and minimal copied metadata for audit/replay.
