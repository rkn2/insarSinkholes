# Eisenhower Deck InSAR-Only Retrospective

This folder now includes a reproducible script to:
- discover SAR/InSAR products over the Eisenhower Parking Deck AOI,
- save manifests for ARIA/OPERA/Sentinel-1,
- run an InSAR-only precursor analysis (change-point + risk score),
- output figures and machine-readable summaries.

## Run

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/insarSinkholes/eisenhower_insar_retrospective.py --max-results 300 --download-browse 20
```

Outputs are written to:

`/Users/rebeccanapolitano/antigravityProjects/digitalTwins/insarSinkholes/outputs/eisenhower_retrospective_upgraded`

## Data Needed For A True (Non-Synthetic) Reconstruction

You only need one measured displacement series for the deck location:

1. `date`
2. `displacement_mm` (InSAR LOS displacement, consistent sign convention)

If you already have InSAR results from MintPy/GMTSAR/ARIA/OPERA, export a CSV with exactly those two columns and run:

```bash
python3 /Users/rebeccanapolitano/antigravityProjects/digitalTwins/insarSinkholes/eisenhower_insar_retrospective.py --insar-csv /absolute/path/to/eisenhower_insar_timeseries.csv
```

## Notes

- The script always performs product discovery; if `--insar-csv` is omitted it uses synthetic InSAR for feasibility mode.
- ARIA/OPERA netCDF downloads from ASF Cumulus generally require Earthdata-authenticated access.
- Browse PNG quicklooks are fetched automatically to support fast visual screening before full download workflows.
