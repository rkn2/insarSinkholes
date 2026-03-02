#!/usr/bin/env python3
"""Build real non-sinkhole control feature rows from OPERA point observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build real control features from OPERA point observations.")
    p.add_argument(
        "--point-observations-csv",
        default="outputs/eisenhower_retrospective/insar_point_observations.csv",
    )
    p.add_argument(
        "--events-csv",
        default="outputs/sinkhole_event_discovery_derived_full/events_used.csv",
    )
    p.add_argument(
        "--source-event-id",
        default="2023-08-16_eisenhower_parking_deck_penn_state",
    )
    p.add_argument("--min-distance-m", type=float, default=300.0)
    p.add_argument("--max-distance-m", type=float, default=1500.0)
    p.add_argument("--n-controls", type=int, default=3)
    p.add_argument("--min-observations", type=int, default=40)
    p.add_argument("--out-csv", default="config/real_controls_features.csv")
    return p.parse_args()


def _safe_series(x: pd.Series, default: float = 0.0) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return y.fillna(default)


def _build_features(g: pd.DataFrame) -> pd.DataFrame:
    out = g.sort_values("date").copy().reset_index(drop=True)
    out["obs_gap_days"] = out["date"].diff().dt.total_seconds().div(86400.0)
    med_gap = float(out["obs_gap_days"].median()) if np.isfinite(out["obs_gap_days"].median()) else 6.0
    out["obs_gap_days"] = out["obs_gap_days"].fillna(med_gap).replace(0.0, np.nan).fillna(med_gap)

    out["cum_settlement_mm"] = out["disp_mm"] - float(out["disp_mm"].iloc[0])
    out["vel_mm_day"] = out["disp_mm"].diff().fillna(0.0) / out["obs_gap_days"]
    out["acc_mm_day2"] = out["vel_mm_day"].diff().fillna(0.0) / out["obs_gap_days"]

    n_base = max(8, len(out) // 4)
    base_v = out["vel_mm_day"].iloc[:n_base]
    base_a = out["acc_mm_day2"].iloc[:n_base]
    mu_v = float(base_v.mean())
    sig_v = float(base_v.std(ddof=0)) + 1e-6
    out["robust_vel_z"] = (mu_v - out["vel_mm_day"]) / sig_v

    mu_a = float((-base_a).mean())
    sig_a = float((-base_a).std(ddof=0)) + 1e-6
    out["robust_accel_z"] = ((-out["acc_mm_day2"]) - mu_a) / sig_a

    cp_th = float(np.quantile(np.abs(out["acc_mm_day2"].to_numpy()), 0.9))
    out["changepoint_flag"] = (np.abs(out["acc_mm_day2"]) >= cp_th).astype(int)
    return out


def main() -> None:
    args = parse_args()
    points = pd.read_csv(args.point_observations_csv)
    events = pd.read_csv(args.events_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.source_event_id not in set(events["event_id"].astype(str)):
        raise RuntimeError(f"source-event-id not found in events csv: {args.source_event_id}")

    event_date = pd.to_datetime(
        events.loc[events["event_id"] == args.source_event_id, "event_date"].iloc[0],
        errors="coerce",
    )
    if pd.isna(event_date):
        raise RuntimeError("Unable to parse source event_date.")

    points["date"] = pd.to_datetime(points["date"], errors="coerce")
    points["disp_mm"] = _safe_series(points.get("disp_m", points.get("short_wavelength_displacement", 0.0))) * 1000.0
    points["dist_m"] = _safe_series(points.get("dist_m", 0.0))
    points["point_id"] = points.get("point_id", "P00").astype(str)
    points = points.dropna(subset=["date", "disp_mm"]).copy()

    cand = points[(points["dist_m"] >= float(args.min_distance_m)) & (points["dist_m"] <= float(args.max_distance_m))].copy()
    if cand.empty:
        raise RuntimeError("No candidate control points in requested distance range.")

    grp = cand.groupby("point_id").agg(
        n_obs=("date", "count"),
        median_abs_disp=("disp_mm", lambda s: float(np.median(np.abs(s.to_numpy())))),
        mean_dist_m=("dist_m", "mean"),
    )
    grp = grp[grp["n_obs"] >= int(args.min_observations)].copy()
    if grp.empty:
        raise RuntimeError("No control points meet min-observations threshold.")

    pick = (
        grp.sort_values(["median_abs_disp", "mean_dist_m", "n_obs"], ascending=[True, True, False])
        .head(int(args.n_controls))
        .reset_index()
    )
    keep_ids = set(pick["point_id"].astype(str))
    kept = cand[cand["point_id"].isin(keep_ids)].copy()

    rows = []
    for pid, g in kept.groupby("point_id"):
        feat = _build_features(g[["date", "disp_mm"]].copy())
        ctrl_id = f"control__{args.source_event_id}__{pid}"
        feat["event_id"] = ctrl_id
        feat["source_event_id"] = args.source_event_id
        feat["dataset"] = "OPERA_S1_DISP"
        feat["split"] = "train"
        feat["event_date"] = event_date
        feat["days_since_start"] = (feat["date"] - feat["date"].min()).dt.total_seconds() / 86400.0
        feat["days_to_event"] = (event_date - feat["date"]).dt.total_seconds() / 86400.0
        feat["is_pre_event_window"] = (feat["days_to_event"] >= 0.0).astype(int)
        feat["risk_label"] = 0
        feat["is_aria"] = 0
        feat["is_opera"] = 1
        feat["disp_source"] = "observed_control"
        rows.append(feat)

    out = pd.concat(rows, ignore_index=True).sort_values(["event_id", "date"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows for {out['event_id'].nunique()} control sites -> {out_csv}")


if __name__ == "__main__":
    main()
