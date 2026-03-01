#!/usr/bin/env python3
"""Retrospective InSAR-only workflow for PSU Eisenhower Parking Deck sinkhole event."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import asf_search as asf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import ruptures as rpt
from geopy.geocoders import Nominatim
from shapely import wkt


EVENT_DATE = pd.Timestamp("2023-08-16")


@dataclass
class Site:
    name: str = "Eisenhower Parking Deck, University Park, PA"
    lat: float = 40.8023011
    lon: float = -77.8609059


def geocode_site(query: str) -> Site:
    geocoder = Nominatim(user_agent="structuralex-insar-retro")
    loc = geocoder.geocode(query, timeout=20)
    if loc is None:
        raise RuntimeError(f"Could not geocode site query: {query}")
    return Site(name=query, lat=float(loc.latitude), lon=float(loc.longitude))


def _asf_results_to_df(results: Iterable, dataset_name: str) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.properties
        browse = p.get("browse")
        browse_url = browse[0] if isinstance(browse, list) and browse else None
        bytes_val = p.get("bytes")
        if isinstance(bytes_val, dict):
            try:
                first_key = next(iter(bytes_val))
                bytes_val = bytes_val[first_key].get("bytes")
            except Exception:
                bytes_val = None
        rows.append(
            {
                "dataset": dataset_name,
                "scene_name": p.get("sceneName"),
                "start_time": p.get("startTime"),
                "stop_time": p.get("stopTime"),
                "url": p.get("url"),
                "browse_url": browse_url,
                "file_name": p.get("fileName"),
                "bytes": bytes_val,
                "path_number": p.get("pathNumber"),
                "frame_number": p.get("frameNumber"),
                "flight_direction": p.get("flightDirection"),
                "orbit": p.get("orbit"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce")
        df = df.sort_values("start_time", ascending=True).reset_index(drop=True)
    return df


def discover_products(
    site: Site, start: str, end: str, max_results: int = 250
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    point_wkt = f"POINT({site.lon} {site.lat})"
    datasets = {
        "ARIA_S1_GUNW": asf.DATASET.ARIA_S1_GUNW,
        "OPERA_S1_DISP": asf.DATASET.OPERA_S1,
        "SENTINEL1_SLC": asf.DATASET.SENTINEL1,
    }
    out: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for name, ds in datasets.items():
        counts[name] = int(
            asf.search_count(
                intersectsWith=point_wkt,
                start=f"{start}T00:00:00Z",
                end=f"{end}T23:59:59Z",
                dataset=ds,
            )
        )
        results = asf.geo_search(
            intersectsWith=point_wkt,
            start=f"{start}T00:00:00Z",
            end=f"{end}T23:59:59Z",
            dataset=ds,
            maxResults=max_results,
        )
        out[name] = _asf_results_to_df(results, name)
    return out, counts


def write_manifest(
    manifests: dict[str, pd.DataFrame], total_counts: dict[str, int], outdir: Path, site: Site
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, df in manifests.items():
        path = outdir / f"manifest_{name.lower()}.csv"
        df.to_csv(path, index=False)
        summary_rows.append(
            {
                "dataset": name,
                "count_downloaded": len(df),
                "count_available": total_counts.get(name),
                "first_scene": df["start_time"].min() if len(df) else pd.NaT,
                "last_scene": df["start_time"].max() if len(df) else pd.NaT,
                "manifest_csv": str(path),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "manifest_summary.csv", index=False)
    metadata = {
        "site_name": site.name,
        "lat": site.lat,
        "lon": site.lon,
        "event_date": str(EVENT_DATE.date()),
    }
    (outdir / "site_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def download_browse_images(manifest: pd.DataFrame, outdir: Path, max_images: int = 25) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    if manifest.empty or "browse_url" not in manifest.columns:
        return 0
    m = manifest.dropna(subset=["browse_url"]).copy()
    if m.empty:
        return 0
    m = m.sort_values("start_time", ascending=False).head(max_images)
    saved = 0
    for _, row in m.iterrows():
        url = row["browse_url"]
        name = str(row["scene_name"]).replace("/", "_")
        ext = ".png" if str(url).lower().endswith(".png") else ".jpg"
        out_path = outdir / f"{name}{ext}"
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200 and r.content:
                out_path.write_bytes(r.content)
                saved += 1
        except Exception:
            continue
    return saved


def synthetic_insar_series(seed: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-02", "2023-08-12", freq="6D")
    t = np.arange(len(dates), dtype=float)
    baseline = -0.015 * t
    pre_event = np.clip((dates - pd.Timestamp("2023-03-01")).days.to_numpy(), 0, None)
    accel = -0.05 * (pre_event / 30.0) ** 1.35
    disp_mm = baseline + accel + rng.normal(0.0, 0.8, size=len(t))
    return pd.DataFrame({"date": dates, "displacement_mm": disp_mm})


def parse_insar_input(
    csv_path: str, site: Site, max_dist_m: float = 1500.0, min_point_obs: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    raw = pd.read_csv(csv_path)
    norm = {c: c.strip().lower().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(columns=norm)

    # Case A: already in simple format.
    if {"date", "displacement_mm"}.issubset(raw.columns):
        out = raw[["date", "displacement_mm"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "displacement_mm"]).sort_values("date").reset_index(drop=True)
        meta = {"input_format": "simple_timeseries", "n_points_used": None}
        return out, None, meta

    # Case B: OPERA/ASF point export format.
    required = {"geometry", "date_(mm/dd/yr)", "short_wavelength_displacement"}
    if not required.issubset(raw.columns):
        raise ValueError(
            "Unsupported CSV format. Expected either columns [date, displacement_mm] "
            "or OPERA export columns [geometry, date (mm/dd/yr), short wavelength displacement]."
        )

    raw["date"] = pd.to_datetime(raw["date_(mm/dd/yr)"], errors="coerce")
    raw["disp_m"] = pd.to_numeric(raw["short_wavelength_displacement"], errors="coerce")
    geom = raw["geometry"].astype(str).apply(wkt.loads)
    raw["lon"] = geom.apply(lambda g: float(g.x))
    raw["lat"] = geom.apply(lambda g: float(g.y))

    # Distance filter keeps points on/around deck while dropping obvious outliers.
    r_earth = 6371000.0
    lat0 = np.deg2rad(site.lat)
    latr = np.deg2rad(raw["lat"].to_numpy())
    dlat = latr - lat0
    dlon = np.deg2rad(raw["lon"].to_numpy() - site.lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(latr) * np.sin(dlon / 2) ** 2
    raw["dist_m"] = 2 * r_earth * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    kept = raw[raw["dist_m"] <= max_dist_m].copy()
    if kept.empty:
        raise ValueError("No InSAR points remained after distance filtering. Increase max_dist_m.")

    # Build point inventory and enforce minimum observation count per point.
    pts = kept[["lon", "lat"]].drop_duplicates().copy().reset_index(drop=True)
    pts["point_id"] = [f"P{i+1:02d}" for i in range(len(pts))]
    kept = kept.merge(pts, on=["lon", "lat"], how="left")
    point_counts = kept.groupby("point_id")["date"].count()
    keep_ids = point_counts[point_counts >= min_point_obs].index
    kept = kept[kept["point_id"].isin(keep_ids)].copy()
    if kept.empty:
        raise ValueError("No points passed min observation filter; reduce --min-point-obs.")

    # Per-date robust outlier rejection using MAD around date median.
    med = kept.groupby("date")["disp_m"].transform("median")
    mad = (kept["disp_m"] - med).abs().groupby(kept["date"]).transform("median") + 1e-9
    robust_z = 0.6745 * (kept["disp_m"] - med).abs() / mad
    kept["robust_z"] = robust_z
    kept = kept[kept["robust_z"] <= 3.5].copy()

    # Aggregate with uncertainty bands across points at each date.
    agg = kept.groupby("date")["disp_m"].agg(
        displacement_m_median="median",
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
        n_points="count",
    )
    agg = agg.reset_index().sort_values("date")
    agg["displacement_mm"] = agg["displacement_m_median"] * 1000.0
    agg["displacement_lo_mm"] = agg["p10"] * 1000.0
    agg["displacement_hi_mm"] = agg["p90"] * 1000.0
    out = agg[["date", "displacement_mm", "displacement_lo_mm", "displacement_hi_mm", "n_points"]].copy().reset_index(drop=True)
    meta = {
        "input_format": "opera_point_export",
        "n_points_total": int(raw[["lon", "lat"]].drop_duplicates().shape[0]),
        "n_points_used": int(kept[["lon", "lat"]].drop_duplicates().shape[0]),
        "distance_filter_m": float(max_dist_m),
        "min_point_obs": int(min_point_obs),
    }
    return out, kept, meta


def analyze_timeseries(
    insar_df: pd.DataFrame,
    event_date: pd.Timestamp,
    claim_end_date: pd.Timestamp,
    false_alarms_per_year: float | None = 1.0,
    fixed_threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = insar_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["displacement_mm"]).reset_index(drop=True)
    if len(df) < 12:
        raise ValueError("Need at least 12 InSAR observations for retrospective analysis.")

    df["smoothed_mm"] = df["displacement_mm"].rolling(3, center=True, min_periods=1).median()
    dt_days = df["date"].diff().dt.days.fillna(df["date"].diff().dt.days.median()).replace(0, np.nan).fillna(6)
    df["velocity_mm_per_day"] = df["smoothed_mm"].diff().fillna(0.0) / dt_days

    # Change-point detection over smoothed displacement.
    y = df["smoothed_mm"].to_numpy().reshape(-1, 1)
    model = rpt.Pelt(model="rbf").fit(y)
    change_idx = [i for i in model.predict(pen=3) if i < len(df)]
    df["changepoint"] = False
    if change_idx:
        df.loc[np.array(change_idx) - 1, "changepoint"] = True

    baseline_end = min(event_date - pd.Timedelta(days=180), df["date"].max())
    baseline = df.loc[df["date"] <= baseline_end, "velocity_mm_per_day"]
    if len(baseline) < 5:
        baseline = df["velocity_mm_per_day"].iloc[: max(5, len(df) // 4)]
    mu = baseline.mean()
    sigma = baseline.std(ddof=0) + 1e-6
    # Negative velocity = settlement; convert to positive risk z.
    df["velocity_risk_z"] = (mu - df["velocity_mm_per_day"]) / sigma
    df["cum_settlement_mm"] = df["smoothed_mm"] - df["smoothed_mm"].iloc[0]
    df["risk_score"] = 0.65 * df["velocity_risk_z"] + 0.35 * np.abs(df["cum_settlement_mm"]) / 8.0

    if fixed_threshold is not None:
        alert_threshold = float(fixed_threshold)
        threshold_method = "fixed"
    else:
        # Calibrate threshold from historical baseline exceedance tolerance.
        if false_alarms_per_year is None or false_alarms_per_year <= 0:
            false_alarms_per_year = 1.0
        baseline_velocity_z = (mu - baseline) / sigma
        baseline_cum = np.abs(df.loc[baseline.index, "cum_settlement_mm"]) / 8.0
        baseline_scores = 0.65 * baseline_velocity_z + 0.35 * baseline_cum
        dt_med = float(dt_days.median()) if np.isfinite(dt_days.median()) else 6.0
        obs_per_year = max(1.0, 365.25 / max(dt_med, 1.0))
        exceed_rate = min(0.49, max(1e-4, float(false_alarms_per_year) / obs_per_year))
        quantile = 1.0 - exceed_rate
        if len(baseline_scores) >= 5:
            alert_threshold = float(np.quantile(baseline_scores.to_numpy(), quantile))
            threshold_method = (
                "historical_false_alarm_calibrated"
                if len(baseline_scores) >= 8
                else "historical_false_alarm_calibrated_low_sample"
            )
        else:
            alert_threshold = 2.2
            threshold_method = "fallback_fixed_insufficient_baseline"
    claim_window = df[df["date"] <= claim_end_date]
    alerts = claim_window[claim_window["risk_score"] >= alert_threshold]
    first_alert = alerts["date"].iloc[0] if not alerts.empty else pd.NaT
    lead_days = int((event_date - first_alert).days) if pd.notna(first_alert) else None
    prehistory_days = int((event_date - df["date"].min()).days)
    prehistory_ok = prehistory_days >= 365

    summary = {
        "n_observations": int(len(df)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "max_abs_cum_settlement_mm": float(np.abs(df["cum_settlement_mm"]).max()),
        "max_velocity_risk_z": float(df["velocity_risk_z"].max()),
        "first_alert_date_in_claim_window": str(first_alert.date()) if pd.notna(first_alert) else None,
        "lead_days_to_event": lead_days,
        "event_date": str(event_date.date()),
        "claim_end_date": str(claim_end_date.date()),
        "alert_threshold": alert_threshold,
        "threshold_method": threshold_method,
        "false_alarms_per_year_target": false_alarms_per_year,
        "prehistory_days": prehistory_days,
        "prehistory_meets_1yr": prehistory_ok,
        "interpretation": (
            "Potential precursor detected before event date."
            if lead_days is not None
            else "No robust precursor crossing threshold before event date."
        ),
    }
    if not prehistory_ok:
        summary["warning"] = "Pre-event history is shorter than 1 year; confidence in precursor conclusions is limited."
    return df, summary


def plot_retrospective(df: pd.DataFrame, summary: dict, out_png: Path, event_date: pd.Timestamp) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if {"displacement_lo_mm", "displacement_hi_mm"}.issubset(df.columns):
        ax1.fill_between(
            df["date"],
            df["displacement_lo_mm"],
            df["displacement_hi_mm"],
            color="tab:blue",
            alpha=0.15,
            label="Point-spread band (10-90%)",
        )
    ax1.plot(df["date"], df["displacement_mm"], "o-", alpha=0.5, label="InSAR displacement (obs)")
    ax1.plot(df["date"], df["smoothed_mm"], "-", lw=2.2, label="Smoothed")
    cp = df[df["changepoint"]]
    if len(cp):
        ax1.scatter(cp["date"], cp["smoothed_mm"], c="red", s=35, label="Detected changepoints")
    ax1.axvline(event_date, color="black", ls="--", lw=1.8, label=f"Sinkhole event ({event_date.date()})")
    ax1.set_ylabel("LOS displacement (mm)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    ax2.plot(df["date"], df["risk_score"], lw=2, color="tab:orange", label="Risk score")
    ax2.axhline(summary["alert_threshold"], color="red", ls="--", label="Alert threshold")
    if summary["first_alert_date_in_claim_window"]:
        ax2.axvline(pd.Timestamp(summary["first_alert_date_in_claim_window"]), color="purple", ls=":")
    ax2.axvline(pd.Timestamp(summary["claim_end_date"]), color="gray", ls=":", label="Claim window end")
    ax2.axvline(event_date, color="black", ls="--")
    ax2.set_ylabel("Dimensionless")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="InSAR-only retrospective for PSU Eisenhower sinkhole case.")
    p.add_argument("--site-query", default="Eisenhower Parking Deck, University Park, PA")
    p.add_argument("--start-date", default="2021-01-01")
    p.add_argument("--end-date", default="2023-08-15")
    p.add_argument("--max-results", type=int, default=250)
    p.add_argument("--insar-csv", default=None, help="Optional CSV with columns: date, displacement_mm")
    p.add_argument("--max-point-distance-m", type=float, default=1500.0, help="Max distance from deck for point-based CSVs.")
    p.add_argument("--min-point-obs", type=int, default=8, help="Minimum observations required per InSAR point.")
    p.add_argument("--event-date", default="2023-08-16", help="Ground-truth event date (YYYY-MM-DD).")
    p.add_argument(
        "--claim-end-date",
        default=None,
        help="Last date allowed for pre-event prediction claims (defaults to event date).",
    )
    p.add_argument(
        "--false-alarms-per-year",
        type=float,
        default=1.0,
        help="Target false alarms per year for threshold calibration; ignored if --fixed-threshold is set.",
    )
    p.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Optional fixed risk threshold override.",
    )
    p.add_argument("--download-browse", type=int, default=20, help="How many browse quicklook images to fetch.")
    p.add_argument("--outdir", default="outputs/eisenhower_retrospective")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    site = geocode_site(args.site_query)
    event_date = pd.Timestamp(args.event_date)
    claim_end_date = pd.Timestamp(args.claim_end_date) if args.claim_end_date else event_date
    analysis_start = pd.Timestamp(args.start_date)
    analysis_end = pd.Timestamp(args.end_date)
    manifests, counts = discover_products(site, args.start_date, args.end_date, max_results=args.max_results)
    write_manifest(manifests, counts, outdir / "data_discovery", site)
    browse_saved = download_browse_images(
        manifests["ARIA_S1_GUNW"], outdir / "data_discovery" / "aria_browse", max_images=args.download_browse
    )

    if args.insar_csv:
        insar, point_obs, input_meta = parse_insar_input(
            args.insar_csv, site, max_dist_m=args.max_point_distance_m, min_point_obs=args.min_point_obs
        )
        source = "provided_csv"
    else:
        insar = synthetic_insar_series(seed=14)
        insar.to_csv(outdir / "synthetic_insar_series.csv", index=False)
        point_obs = None
        input_meta = {"input_format": "synthetic_demo"}
        source = "synthetic_demo"

    # Apply analysis window to both aggregate and point observations.
    insar["date"] = pd.to_datetime(insar["date"], errors="coerce")
    insar = insar[(insar["date"] >= analysis_start) & (insar["date"] <= analysis_end)].copy()
    if point_obs is not None and "date" in point_obs.columns:
        point_obs["date"] = pd.to_datetime(point_obs["date"], errors="coerce")
        point_obs = point_obs[(point_obs["date"] >= analysis_start) & (point_obs["date"] <= analysis_end)].copy()

    analyzed, summary = analyze_timeseries(
        insar,
        event_date=event_date,
        claim_end_date=claim_end_date,
        false_alarms_per_year=args.false_alarms_per_year,
        fixed_threshold=args.fixed_threshold,
    )
    summary["source"] = source
    summary.update(input_meta)
    summary["analysis_start_date"] = str(analysis_start.date())
    summary["analysis_end_date"] = str(analysis_end.date())
    analyzed.to_csv(outdir / "insar_retrospective_timeseries.csv", index=False)
    if point_obs is not None:
        point_obs.to_csv(outdir / "insar_point_observations.csv", index=False)
    (outdir / "retrospective_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_retrospective(analyzed, summary, outdir / "retrospective_plot.png", event_date=event_date)
    (outdir / "input_metadata.json").write_text(json.dumps(input_meta, indent=2), encoding="utf-8")

    print("Retrospective run complete.")
    print(f"Site: {site.name} ({site.lat:.6f}, {site.lon:.6f})")
    for name, df in manifests.items():
        print(f"{name}: {len(df)} products fetched in manifest ({counts[name]} total available)")
    print(f"ARIA browse quicklooks downloaded: {browse_saved}")
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
