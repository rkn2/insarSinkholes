# InSAR Sinkhole Retrospective + Digital Twin Dashboard

This repository contains a practical workflow for retrospective sinkhole risk analysis around the **Eisenhower Parking Deck (Penn State)** using InSAR point data, plus a Streamlit dashboard for visualization.

## What This Project Does

- Discovers relevant SAR/InSAR products (ARIA, OPERA, Sentinel-1) for a site and date window.
- Ingests InSAR time series from:
  - a simple CSV (`date`, `displacement_mm`), or
  - OPERA/ASF point export CSV (`geometry`, `date (mm/dd/yr)`, `short wavelength displacement`).
- Filters and aggregates point observations with robust statistics.
- Computes a precursor risk score from displacement trend + cumulative settlement + non-stationary acceleration.
- Fits an inverted Gaussian subsidence bowl to per-date point clouds and adds bowl-fit diagnostics to risk scoring.
- Runs a statistical slope-break ("kink") test and reports break date + significance.
- Calibrates alert threshold from historical false-alarm tolerance (instead of fixed only).
- Produces a retrospective summary and plots.
- Serves an interactive dashboard with:
  - plan view + section view,
  - time series + risk score,
  - 3D point-cloud projection with interpolated InSAR displacement.

---

## Repository Structure

- `eisenhower_insar_retrospective.py`  
  Main retrospective analysis pipeline.
- `settlement_dashboard.py`  
  Streamlit app for visualization.
- `synthetic_structural_twin_demo.py`  
  Synthetic demo generator (InSAR + accelerometer fusion).
- `export_settlement_dashboard_html.py`  
  Exports a static HTML dashboard snapshot.
- `EISENHOWER_RETROSPECTIVE.md`  
  Focused project notes.
- `outputs/`  
  Generated outputs (CSV, JSON, PNG, HTML, manifests).

---

## Quick Start

## 1) Install dependencies

```bash
python3 -m pip install pandas numpy scipy matplotlib streamlit plotly asf_search geopy shapely osmnx pyproj ruptures
```

## 2) Run retrospective analysis

Example using OPERA point export CSV:

```bash
python3 eisenhower_insar_retrospective.py \
  --start-date 2020-01-01 \
  --end-date 2026-12-31 \
  --event-date 2023-08-16 \
  --claim-end-date 2023-08-16 \
  --false-alarms-per-year 1 \
  --min-point-obs 20 \
  --insar-csv outputs/eisenhower_retrospective/asf-opera-displacement-2026-02-28_09-27-45.csv \
  --outdir outputs/eisenhower_retrospective_upgraded
```

Outputs are written to:

- `outputs/eisenhower_retrospective_upgraded/insar_retrospective_timeseries.csv`
- `outputs/eisenhower_retrospective_upgraded/retrospective_summary.json`
- `outputs/eisenhower_retrospective_upgraded/retrospective_plot.png`
- `outputs/eisenhower_retrospective_upgraded/insar_point_observations.csv` (for point-based input)

## 3) Launch dashboard

```bash
streamlit run settlement_dashboard.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).
The app defaults to the Eisenhower retrospective mode when retrospective outputs are available.

---

## Input Data Formats

## A) Simple time-series CSV

Required columns:

- `date`
- `displacement_mm`

## B) OPERA/ASF point export CSV

Expected columns:

- `geometry` (WKT point, e.g., `POINT(lon lat)`)
- `date (mm/dd/yr)`
- `short wavelength displacement` (meters)

The pipeline:

- parses WKT into point coordinates,
- filters by distance to site,
- enforces minimum point observations (`--min-point-obs`),
- rejects per-date outliers using robust MAD filtering,
- aggregates by date using median and quantile uncertainty bands.

---

## Risk Score and Threshold

Risk score combines:

- normalized settlement velocity (`velocity_risk_z`)
- cumulative settlement magnitude
- non-stationary acceleration (`accel_risk_z`) from short/long velocity mismatch
- Gaussian bowl evidence (`gaussian_bowl_risk`) weighted by fit quality

Default formula:

- `risk_score = 0.45*velocity_risk_z + 0.30*abs(cum_settlement_mm)/8.0 + 0.15*clip(accel_risk_z,0,∞) + 0.10*clip(gaussian_bowl_risk,0,∞)`

Threshold options:

- **Calibrated** (default): derived from baseline history using `--false-alarms-per-year`.
- **Fixed**: override with `--fixed-threshold`.

Important: if pre-event history is short, threshold confidence is lower; this is reported in summary JSON.

---

## Optional: True Sinkhole Marker on Plan View

The dashboard can place a real sinkhole marker if this file exists:

- `outputs/eisenhower_retrospective_upgraded/sinkhole_location.csv`

CSV format (single row):

```csv
lat,lon
40.8023,-77.8609
```

If absent, the dashboard uses an inferred location from public descriptions.

---

## Notes and Limitations

- InSAR-only models are useful for trend detection, not deterministic failure prediction.
- Utilities/drainage failures can trigger rapid local failures that InSAR may only partially capture.
- Use this workflow with engineering judgment and supplemental site information (utilities, inspections, repairs, geotech context).
- LOS decomposition (ascending + descending orbit fusion) is not yet implemented.
- Tropospheric correction inputs (e.g., GACOS + GNSS) are not yet integrated.
- Deep learning segmentation (UNet/LSTM) is not yet integrated in this repository.
- Multispectral stress proxies (NDVI/MI) are not yet fused into the current risk score.

---

## Suggested Next Improvements

- Add ascending/descending orbit LOS decomposition for vertical/horizontal motion separation.
- Integrate atmospheric correction fields (GACOS or equivalent) before time-series analysis.
- Add coherence/quality weighting per point (PSI-like persistence weighting).
- Add multispectral hydro-stress proxy fusion (NDVI/MI/backscatter) for vegetation-heavy areas.
- Add optional UNet/LSTM module for automated interferogram/time-series anomaly segmentation.
