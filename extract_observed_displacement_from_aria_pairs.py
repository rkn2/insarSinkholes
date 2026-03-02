#!/usr/bin/env python3
"""Extract observed displacement from ARIA GUNW .nc files using pair-aware dates.

For each ARIA interferogram file:
- sample displacement at event location,
- parse master/slave dates from filename,
- compute observation date as midpoint(master, slave).

Writes:
1) raw rows (one per file): event_id,dataset,master_date,slave_date,date,disp_mm,...
2) aggregated rows (per event_id,dataset,date): median disp_mm + pair count.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair-aware ARIA displacement extraction.")
    p.add_argument(
        "--download-outdir",
        action="append",
        required=True,
        help="Root with per-event downloads/aria_s1_gunw folders. Repeat for multiple roots.",
    )
    p.add_argument("--manifest-root", required=True, help="Root containing events_used.csv with event_id/lat/lon.")
    p.add_argument("--out-csv-raw", required=True)
    p.add_argument("--out-csv-agg", required=True)
    p.add_argument("--max-files-per-event", type=int, default=500)
    return p.parse_args()


def _event_map(manifest_root: Path) -> dict[str, dict]:
    events_file = manifest_root / "events_used.csv"
    if not events_file.exists():
        raise FileNotFoundError(f"Missing {events_file}")
    ev = pd.read_csv(events_file)
    req = {"event_id", "latitude", "longitude", "event_date"}
    if not req.issubset(ev.columns):
        raise ValueError(f"events_used.csv missing columns: {sorted(req - set(ev.columns))}")
    return {
        str(r["event_id"]): {
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "event_date": str(r["event_date"]),
        }
        for _, r in ev.iterrows()
    }


def _parse_pair_dates(filename: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    m = re.search(r"(20\d{6})_(20\d{6})", filename)
    if not m:
        return None, None
    d1 = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
    d2 = pd.to_datetime(m.group(2), format="%Y%m%d", errors="coerce")
    if pd.isna(d1) or pd.isna(d2):
        return None, None
    return d1, d2


def _read_point_disp_mm(nc_path: Path, lat: float, lon: float) -> float | None:
    import h5py

    with h5py.File(nc_path, "r") as f:
        cand_data = [
            "science/grids/data/displacement",
            "science/grids/data/short_wavelength_displacement",
            "science/grids/data/unwrappedPhase",
            "displacement",
            "short_wavelength_displacement",
            "unwrappedPhase",
        ]
        data_key = next((k for k in cand_data if k in f), None)
        if data_key is None:
            return None

        data = f[data_key][()]
        data = np.squeeze(data)
        if data.ndim != 2:
            return None

        lat_key = next((k for k in ["science/grids/data/latitude", "latitude", "lat"] if k in f), None)
        lon_key = next((k for k in ["science/grids/data/longitude", "longitude", "lon"] if k in f), None)

        if lat_key and lon_key:
            lat_arr = f[lat_key][()]
            lon_arr = f[lon_key][()]
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                iy = int(np.argmin(np.abs(lat_arr - lat)))
                ix = int(np.argmin(np.abs(lon_arr - lon)))
            else:
                d2 = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
                iy, ix = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
        else:
            attrs = dict(f[data_key].attrs)
            x_first = attrs.get("x_first")
            y_first = attrs.get("y_first")
            x_step = attrs.get("x_step")
            y_step = attrs.get("y_step")
            if any(v is None for v in [x_first, y_first, x_step, y_step]):
                return None
            ix = int(round((lon - float(x_first)) / float(x_step)))
            iy = int(round((lat - float(y_first)) / float(y_step)))

        iy = max(0, min(int(iy), data.shape[0] - 1))
        ix = max(0, min(int(ix), data.shape[1] - 1))
        val = float(data[iy, ix])
        if not np.isfinite(val):
            return None

        if "phase" in data_key.lower():
            val = val * 0.0555 / (4.0 * math.pi) * 1000.0
        else:
            if abs(val) < 5.0:
                val = val * 1000.0

        return float(val)


def main() -> None:
    args = parse_args()

    manifest_root = Path(args.manifest_root)
    out_raw = Path(args.out_csv_raw)
    out_agg = Path(args.out_csv_agg)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_agg.parent.mkdir(parents=True, exist_ok=True)

    event_meta = _event_map(manifest_root)

    rows = []
    for root_str in args.download_outdir:
        root = Path(root_str)
        for event_id, meta in sorted(event_meta.items()):
            aria_dir = root / event_id / "downloads" / "aria_s1_gunw"
            if not aria_dir.exists():
                continue
            files = sorted(aria_dir.glob("*.nc"))[: int(args.max_files_per_event)]
            for fp in files:
                d1, d2 = _parse_pair_dates(fp.name)
                if d1 is None or d2 is None:
                    continue
                obs_date = d1 + (d2 - d1) / 2
                try:
                    disp_mm = _read_point_disp_mm(fp, lat=meta["lat"], lon=meta["lon"])
                except Exception:
                    continue
                if disp_mm is None or not np.isfinite(disp_mm):
                    continue
                rows.append(
                    {
                        "event_id": event_id,
                        "dataset": "ARIA_S1_GUNW",
                        "master_date": d1.date().isoformat(),
                        "slave_date": d2.date().isoformat(),
                        "date": pd.Timestamp(obs_date).date().isoformat(),
                        "disp_mm": float(disp_mm),
                        "lat": meta["lat"],
                        "lon": meta["lon"],
                        "source_root": str(root),
                        "source_file": str(fp),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raw.to_csv(out_raw, index=False)
        pd.DataFrame(columns=["event_id", "dataset", "date", "disp_mm", "n_pairs"]).to_csv(out_agg, index=False)
        print("No rows extracted.")
        print(f"raw: {out_raw}")
        print(f"agg: {out_agg}")
        return

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "disp_mm"]).sort_values(["event_id", "dataset", "date", "source_file"]).reset_index(drop=True)
    raw["date"] = raw["date"].dt.date.astype(str)
    raw.to_csv(out_raw, index=False)

    agg = raw.groupby(["event_id", "dataset", "date"], as_index=False).agg(
        disp_mm=("disp_mm", "median"),
        n_pairs=("disp_mm", "count"),
    )
    agg = agg.sort_values(["event_id", "dataset", "date"]).reset_index(drop=True)
    agg.to_csv(out_agg, index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "agg_rows": int(len(agg)),
        "events": int(agg["event_id"].nunique()),
        "raw_out": str(out_raw.resolve()),
        "agg_out": str(out_agg.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
