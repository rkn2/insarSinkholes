#!/usr/bin/env python3
"""Extract observed displacement time series from downloaded ARIA GUNW .nc files.

Reads event manifests and event coordinates from sinkhole_asf_discovery defaults,
samples nearest valid displacement pixel at each event location, and writes:
[event_id,dataset,date,disp_mm,source_file,lat,lon]
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
    p = argparse.ArgumentParser(description="Extract observed displacement from ARIA GUNW netCDF files.")
    p.add_argument("--download-outdir", default="/scratch/rjn5308/sinkholes/outputs/sinkhole_event_discovery")
    p.add_argument("--manifest-root", default="outputs/sinkhole_event_discovery_derived_full")
    p.add_argument("--out-csv", default="config/observed_displacement_aria.csv")
    p.add_argument("--max-files-per-event", type=int, default=250)
    return p.parse_args()


def _event_map_from_manifests(manifest_root: Path) -> dict[str, dict]:
    events_file = manifest_root / "events_used.csv"
    if not events_file.exists():
        raise FileNotFoundError(f"Missing {events_file}")
    ev = pd.read_csv(events_file)
    if "event_id" not in ev.columns:
        raise ValueError("events_used.csv must contain event_id")
    return {
        str(r["event_id"]): {
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "event_date": str(r["event_date"]),
        }
        for _, r in ev.iterrows()
    }


def _read_netcdf_point(nc_path: Path, lat: float, lon: float) -> float | None:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("h5py is required to read ARIA .nc files") from exc

    with h5py.File(nc_path, "r") as f:
        # ARIA typically stores displacement in science/grids/data/unwrappedPhase
        # or displacement grids. We prioritize displacement-like fields.
        cands = [
            "science/grids/data/displacement",
            "science/grids/data/short_wavelength_displacement",
            "science/grids/data/unwrappedPhase",
            "displacement",
            "short_wavelength_displacement",
            "unwrappedPhase",
        ]
        data_key = None
        for k in cands:
            if k in f:
                data_key = k
                break
        if data_key is None:
            # try to discover first 2D float dataset
            for k in f.keys():
                obj = f[k]
                if hasattr(obj, "shape") and len(obj.shape) >= 2:
                    data_key = k
                    break
        if data_key is None:
            return None

        data = f[data_key][()]
        if data.ndim > 2:
            data = np.squeeze(data)
        if data.ndim != 2:
            return None

        # Try coordinate arrays first.
        lat_key = None
        lon_key = None
        for k in ["science/grids/data/latitude", "latitude", "lat"]:
            if k in f:
                lat_key = k
                break
        for k in ["science/grids/data/longitude", "longitude", "lon"]:
            if k in f:
                lon_key = k
                break

        if lat_key and lon_key:
            lat_arr = f[lat_key][()]
            lon_arr = f[lon_key][()]
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                iy = int(np.argmin(np.abs(lat_arr - lat)))
                ix = int(np.argmin(np.abs(lon_arr - lon)))
            else:
                # 2D lat/lon arrays
                d2 = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
                iy, ix = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
        else:
            # Fallback geotransform-like attrs.
            attrs = dict(f[data_key].attrs)
            x_first = attrs.get("x_first")
            y_first = attrs.get("y_first")
            x_step = attrs.get("x_step")
            y_step = attrs.get("y_step")
            if x_first is None or y_first is None or x_step is None or y_step is None:
                # Can't geolocate robustly.
                return None
            ix = int(round((lon - float(x_first)) / float(x_step)))
            iy = int(round((lat - float(y_first)) / float(y_step)))

        iy = max(0, min(int(iy), data.shape[0] - 1))
        ix = max(0, min(int(ix), data.shape[1] - 1))
        val = float(data[iy, ix])
        if not np.isfinite(val):
            return None

        # Convert if likely radians (unwrappedPhase); approximate LOS mm conversion.
        # Sentinel-1 lambda ~ 0.0555 m; displacement = phase * lambda / (4*pi)
        if "phase" in data_key.lower():
            val = val * 0.0555 / (4.0 * math.pi) * 1000.0
        else:
            # If in meters, convert to mm when magnitude looks meter-scale.
            if abs(val) < 5.0:
                val = val * 1000.0

        return float(val)


def _date_from_filename(name: str) -> str | None:
    # ARIA names include pairs like 20240115_20230707; we use the first date as observation date.
    m = re.search(r"(20\d{6})_(20\d{6})", name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce").date().isoformat()


def main() -> None:
    args = parse_args()
    download_outdir = Path(args.download_outdir)
    manifest_root = Path(args.manifest_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    event_map = _event_map_from_manifests(manifest_root)
    rows = []

    for event_id, meta in sorted(event_map.items()):
        aria_dir = download_outdir / event_id / "downloads" / "aria_s1_gunw"
        if not aria_dir.exists():
            continue
        files = sorted(aria_dir.glob("*.nc"))[: int(args.max_files_per_event)]
        for fp in files:
            obs_date = _date_from_filename(fp.name)
            if obs_date is None:
                continue
            try:
                disp_mm = _read_netcdf_point(fp, lat=meta["lat"], lon=meta["lon"])
            except Exception:
                continue
            if disp_mm is None or not np.isfinite(disp_mm):
                continue
            rows.append(
                {
                    "event_id": event_id,
                    "dataset": "ARIA_S1_GUNW",
                    "date": obs_date,
                    "disp_mm": float(disp_mm),
                    "source_file": str(fp),
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(out_csv, index=False)
        print("No ARIA observed displacement rows extracted.")
        print(f"Wrote empty file: {out_csv.resolve()}")
        return

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "disp_mm"])
    out = out.sort_values(["event_id", "dataset", "date"]).reset_index(drop=True)
    out["date"] = out["date"].dt.date.astype(str)

    out.to_csv(out_csv, index=False)

    summary = {
        "rows": int(len(out)),
        "events": int(out["event_id"].nunique()),
        "datasets": sorted(out["dataset"].unique().tolist()),
        "out_csv": str(out_csv.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
