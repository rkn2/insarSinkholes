#!/usr/bin/env python3
"""Train a baseline precursor risk model from sinkhole discovery outputs.

Supports:
- Manifest-based proxy displacement features.
- Optional observed displacement ingestion override (event_id,dataset,date,disp_mm).
- Tighter risk labels requiring both temporal proximity and dynamic precursor evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _slug(text: str) -> str:
    keep = []
    for c in str(text).lower().strip():
        if c.isalnum():
            keep.append(c)
        elif c in {" ", "-", ",", "/"}:
            keep.append("_")
    raw = "".join(keep)
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_")[:80]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline sinkhole precursor risk model.")
    p.add_argument("--discovery-outdir", default="outputs/sinkhole_event_discovery")
    p.add_argument("--split-file", default="config/event_split.yaml")
    p.add_argument("--outdir", default="outputs/ml")
    p.add_argument("--datasets", nargs="+", default=["ARIA_S1_GUNW", "OPERA_S1_DISP"])
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument(
        "--positive-class-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for positive labels during model fitting.",
    )

    # Tighter label defaults
    p.add_argument("--label-window-days", type=int, default=30)
    p.add_argument("--min-vel-z", type=float, default=0.5)
    p.add_argument("--min-accel-z", type=float, default=0.25)

    # Optional observed displacement ingest
    p.add_argument(
        "--observed-displacement-csv",
        default=None,
        help="Optional CSV with columns: event_id,dataset,date,disp_mm to override proxy disp_mm when available.",
    )
    return p.parse_args()


def load_split(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = {
        "train": sorted(str(x) for x in data.get("train", [])),
        "val": sorted(str(x) for x in data.get("val", [])),
        "test": sorted(str(x) for x in data.get("test", [])),
    }
    return out


def _load_manifest_table(discovery_outdir: Path) -> pd.DataFrame:
    all_path = discovery_outdir / "all_manifests.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing all_manifests.csv: {all_path}")
    df = pd.read_csv(all_path)
    if "event_id" not in df.columns:
        if {"event_date", "location"}.issubset(df.columns):
            df["event_id"] = df.apply(
                lambda r: f"{pd.Timestamp(r['event_date']).date()}_{_slug(r['location'])}",
                axis=1,
            )
        else:
            raise ValueError("all_manifests.csv must include event_id or [event_date, location] columns.")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    return df


def _load_events(discovery_outdir: Path) -> pd.DataFrame:
    p = discovery_outdir / "events_used.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing events_used.csv: {p}")
    events = pd.read_csv(p)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
    if "event_id" not in events.columns:
        if "location" not in events.columns:
            raise ValueError("events_used.csv must include event_id or location.")
        events["event_id"] = events.apply(
            lambda r: f"{pd.Timestamp(r['event_date']).date()}_{_slug(r['location'])}",
            axis=1,
        )
    return events


def _load_observed_displacement(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Observed displacement CSV not found: {p}")
    df = pd.read_csv(p)
    required = {"event_id", "dataset", "date", "disp_mm"}
    if not required.issubset(df.columns):
        raise ValueError(f"Observed displacement CSV must include {sorted(required)}")
    df = df[list(required)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["disp_mm"] = pd.to_numeric(df["disp_mm"], errors="coerce")
    df = df.dropna(subset=["date", "disp_mm"]).sort_values(["event_id", "dataset", "date"]).reset_index(drop=True)
    return df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _inject_observed_disp(ts: pd.DataFrame, event_id: str, dataset: str, observed_df: pd.DataFrame | None) -> pd.DataFrame:
    if observed_df is None:
        ts["disp_source"] = "proxy"
        return ts

    sub = observed_df[(observed_df["event_id"] == event_id) & (observed_df["dataset"] == dataset)].copy()
    if sub.empty:
        ts["disp_source"] = "proxy"
        return ts

    # Interpolate observed displacement onto manifest timeline (nearest with tolerance).
    sub = sub.sort_values("date")
    left = ts[["date"]].copy().sort_values("date")
    merged = pd.merge_asof(
        left,
        sub[["date", "disp_mm"]].sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=18),
    )

    out = ts.copy().sort_values("date").reset_index(drop=True)
    obs = merged["disp_mm"].to_numpy()
    mask = ~pd.isna(obs)
    if mask.any():
        out.loc[mask, "disp_mm"] = obs[mask]
        out["disp_source"] = np.where(mask, "observed", "proxy")
    else:
        out["disp_source"] = "proxy"
    return out


def _build_feature_rows(
    manifest_df: pd.DataFrame,
    events_df: pd.DataFrame,
    split: dict[str, list[str]],
    label_window_days: int,
    min_vel_z: float,
    min_accel_z: float,
    observed_df: pd.DataFrame | None,
) -> pd.DataFrame:
    split_map = {}
    for s in ["train", "val", "test"]:
        for eid in split.get(s, []):
            split_map[eid] = s

    events_map = events_df.set_index("event_id")["event_date"].to_dict()
    rows = []

    for (event_id, dataset), g in manifest_df.groupby(["event_id", "dataset"]):
        if event_id not in events_map:
            continue
        event_date = events_map[event_id]

        ts = (
            g[["start_time"]]
            .dropna()
            .drop_duplicates()
            .sort_values("start_time")
            .rename(columns={"start_time": "date"})
            .reset_index(drop=True)
        )
        if ts.empty:
            continue

        ts["event_id"] = event_id
        ts["dataset"] = dataset
        ts["split"] = split_map.get(event_id, "train")
        ts["event_date"] = event_date

        ts["days_since_start"] = (ts["date"] - ts["date"].min()).dt.total_seconds() / 86400.0
        ts["days_to_event"] = (event_date - ts["date"]).dt.total_seconds() / 86400.0
        ts["is_pre_event_window"] = ts["days_to_event"] >= 0.0

        # Proxy displacement baseline from temporal structure.
        pre_proximity = np.clip((180.0 - ts["days_to_event"].to_numpy()) / 180.0, 0.0, 1.0)
        ds_bias = -0.6 if dataset == "ARIA_S1_GUNW" else -0.3
        seasonal = 0.20 * np.sin(np.arange(len(ts)) / 3.0)
        ts["disp_mm"] = (
            -0.018 * ts["days_since_start"].to_numpy()
            + ds_bias
            - 4.2 * (pre_proximity**2)
            + seasonal
        )

        # Optional observed displacement override.
        ts = _inject_observed_disp(ts, event_id=event_id, dataset=dataset, observed_df=observed_df)

        delta_days = ts["date"].diff().dt.total_seconds().div(86400.0)
        delta_days = delta_days.fillna(delta_days.median()).replace(0, np.nan).fillna(6.0)
        ts["obs_gap_days"] = delta_days
        ts["cum_settlement_mm"] = ts["disp_mm"] - float(ts["disp_mm"].iloc[0])
        ts["vel_mm_day"] = ts["disp_mm"].diff().fillna(0.0) / ts["obs_gap_days"]
        ts["acc_mm_day2"] = ts["vel_mm_day"].diff().fillna(0.0) / ts["obs_gap_days"]

        baseline_v = ts.loc[ts["days_to_event"] > 180.0, "vel_mm_day"]
        baseline_a = ts.loc[ts["days_to_event"] > 180.0, "acc_mm_day2"]
        if len(baseline_v) < 5:
            baseline_v = ts["vel_mm_day"].iloc[: max(5, len(ts) // 4)]
        if len(baseline_a) < 5:
            baseline_a = ts["acc_mm_day2"].iloc[: max(5, len(ts) // 4)]

        mu_v = float(baseline_v.mean())
        sigma_v = float(baseline_v.std(ddof=0)) + 1e-6
        ts["robust_vel_z"] = (mu_v - ts["vel_mm_day"]) / sigma_v

        # More negative acceleration (toward stronger settlement) increases risk.
        accel_risk = -ts["acc_mm_day2"]
        mu_a = float((-baseline_a).mean())
        sigma_a = float((-baseline_a).std(ddof=0)) + 1e-6
        ts["robust_accel_z"] = (accel_risk - mu_a) / sigma_a

        cp_threshold = float(np.quantile(np.abs(ts["acc_mm_day2"].to_numpy()), 0.9))
        ts["changepoint_flag"] = (np.abs(ts["acc_mm_day2"]) >= cp_threshold).astype(int)

        ts["is_aria"] = int(dataset == "ARIA_S1_GUNW")
        ts["is_opera"] = int(dataset == "OPERA_S1_DISP")

        # Tightened labels: event window + velocity + acceleration precursor signals.
        ts["risk_label"] = (
            (ts["is_pre_event_window"]) 
            & (ts["days_to_event"] <= float(label_window_days))
            & (ts["days_to_event"] >= 0.0)
            & (ts["robust_vel_z"] >= float(min_vel_z))
            & (ts["robust_accel_z"] >= float(min_accel_z))
        ).astype(int)

        rows.append(ts)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["days_to_event"] = out["days_to_event"].astype(float)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values(["event_id", "dataset", "date"]).reset_index(drop=True)
    return out


def _fit_linear_risk_model(
    train_df: pd.DataFrame, ridge: float, positive_class_weight: float
) -> tuple[np.ndarray, list[str], dict[str, list[float]]]:
    feat_cols = [
        "cum_settlement_mm",
        "vel_mm_day",
        "acc_mm_day2",
        "robust_vel_z",
        "robust_accel_z",
        "changepoint_flag",
        "obs_gap_days",
        "is_aria",
        "is_opera",
    ]
    X = train_df[feat_cols].to_numpy(dtype=float)
    y = train_df["risk_label"].to_numpy(dtype=float)
    w = np.where(y > 0.5, float(positive_class_weight), 1.0).astype(float)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma

    Xb = np.c_[np.ones(len(Xn)), Xn]
    lam = float(ridge)
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0
    sw = np.sqrt(w)
    Xw = Xb * sw[:, None]
    yw = y * sw
    beta = np.linalg.solve(Xw.T @ Xw + lam * I, Xw.T @ yw)

    scaler = {"mu": mu.tolist(), "sigma": sigma.tolist()}
    return beta, feat_cols, scaler


def _predict_proba(df: pd.DataFrame, beta: np.ndarray, feat_cols: list[str], scaler: dict[str, list[float]]) -> np.ndarray:
    X = df[feat_cols].to_numpy(dtype=float)
    mu = np.array(scaler["mu"], dtype=float)
    sigma = np.array(scaler["sigma"], dtype=float)
    Xn = (X - mu) / sigma
    Xb = np.c_[np.ones(len(Xn)), Xn]
    return _sigmoid(Xb @ beta)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _false_alarms_per_year(df: pd.DataFrame, pred_col: str) -> float:
    pre = df[df["is_pre_event_window"]].copy()
    if pre.empty:
        return 0.0
    non_target = pre[pre["risk_label"] == 0]
    if non_target.empty:
        return 0.0
    years = max(1e-6, len(non_target) / 52.0)
    false_alerts = float((non_target[pred_col] == 1).sum())
    return false_alerts / years


def _lead_time_summary(df: pd.DataFrame, pred_col: str) -> dict[str, float | None]:
    lead_days = []
    for _, g in df.groupby("event_id"):
        g = g.sort_values("date")
        alerts = g[(g[pred_col] == 1) & (g["is_pre_event_window"])].copy()
        if alerts.empty:
            continue
        first_alert = g.loc[alerts.index[0], "date"]
        event_date = g["event_date"].iloc[0]
        lead_days.append(float((event_date - first_alert).total_seconds() / 86400.0))
    if not lead_days:
        return {"median_lead_days": None, "mean_lead_days": None, "events_with_alert": 0}
    return {
        "median_lead_days": float(np.median(lead_days)),
        "mean_lead_days": float(np.mean(lead_days)),
        "events_with_alert": int(len(lead_days)),
    }


def _calibration_summary(df: pd.DataFrame, prob_col: str) -> list[dict]:
    d = df.copy()
    d["bin"] = pd.cut(d[prob_col], bins=np.linspace(0, 1, 6), include_lowest=True)
    rows = []
    for k, g in d.groupby("bin", observed=False):
        if len(g) == 0:
            continue
        rows.append(
            {
                "bin": str(k),
                "n": int(len(g)),
                "mean_pred": float(g[prob_col].mean()),
                "mean_obs": float(g["risk_label"].mean()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    discovery_outdir = Path(args.discovery_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    split = load_split(Path(args.split_file))
    manifests = _load_manifest_table(discovery_outdir)
    events = _load_events(discovery_outdir)
    observed = _load_observed_displacement(args.observed_displacement_csv)

    manifests = manifests[manifests["dataset"].isin(args.datasets)].copy()
    if manifests.empty:
        raise RuntimeError("No rows found for selected datasets in all_manifests.csv.")

    feat = _build_feature_rows(
        manifests,
        events,
        split,
        label_window_days=args.label_window_days,
        min_vel_z=args.min_vel_z,
        min_accel_z=args.min_accel_z,
        observed_df=observed,
    )
    if feat.empty:
        raise RuntimeError("Feature extraction produced zero rows.")

    required_cols = [
        "event_id",
        "date",
        "dataset",
        "split",
        "disp_mm",
        "vel_mm_day",
        "acc_mm_day2",
        "risk_label",
        "is_pre_event_window",
        "days_to_event",
    ]
    for c in required_cols:
        if c not in feat.columns:
            raise RuntimeError(f"Missing required feature column: {c}")

    feat = feat.sort_values(["event_id", "dataset", "date"]).reset_index(drop=True)
    feat.to_parquet(outdir / "feature_table.parquet", index=False)
    feat.to_csv(outdir / "feature_table.csv", index=False)

    pre = feat[feat["is_pre_event_window"]].copy()
    train_df = pre[pre["split"] == "train"].copy()
    val_df = pre[pre["split"] == "val"].copy()
    test_df = pre[pre["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Train/val/test splits must all have at least one pre-event row.")

    beta, feat_cols, scaler = _fit_linear_risk_model(
        train_df,
        ridge=args.ridge,
        positive_class_weight=args.positive_class_weight,
    )

    for dset in [train_df, val_df, test_df]:
        dset["risk_prob"] = _predict_proba(dset, beta, feat_cols, scaler)

    best_threshold = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.1, 0.9, 33):
        pred = (val_df["risk_prob"].to_numpy() >= th).astype(int)
        f1 = _binary_metrics(val_df["risk_label"].to_numpy(), pred)["f1"]
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(th)

    for dset in [train_df, val_df, test_df]:
        dset["risk_pred"] = (dset["risk_prob"] >= best_threshold).astype(int)

    metrics = {
        "threshold_tuned_on_val": best_threshold,
        "train": _binary_metrics(train_df["risk_label"].to_numpy(), train_df["risk_pred"].to_numpy()),
        "val": _binary_metrics(val_df["risk_label"].to_numpy(), val_df["risk_pred"].to_numpy()),
        "test": _binary_metrics(test_df["risk_label"].to_numpy(), test_df["risk_pred"].to_numpy()),
        "test_false_alarms_per_year": _false_alarms_per_year(test_df, "risk_pred"),
        "test_lead_time": _lead_time_summary(test_df, "risk_pred"),
        "test_calibration": _calibration_summary(test_df, "risk_prob"),
    }

    training_artifact = {
        "feature_columns": feat_cols,
        "scaler": scaler,
        "beta": beta.tolist(),
        "datasets": list(args.datasets),
        "label_window_days": int(args.label_window_days),
        "label_min_vel_z": float(args.min_vel_z),
        "label_min_accel_z": float(args.min_accel_z),
        "positive_class_weight": float(args.positive_class_weight),
        "observed_displacement_csv": args.observed_displacement_csv,
        "observed_points_used": int((feat.get("disp_source", pd.Series(dtype=str)) == "observed").sum()) if "disp_source" in feat.columns else 0,
        "split_counts": {
            "train_events": len(split.get("train", [])),
            "val_events": len(split.get("val", [])),
            "test_events": len(split.get("test", [])),
        },
        "holdout_events": split.get("test", []),
    }

    (outdir / "model_artifact.json").write_text(json.dumps(training_artifact, indent=2), encoding="utf-8")
    (outdir / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    train_df.to_csv(outdir / "train_predictions.csv", index=False)
    val_df.to_csv(outdir / "val_predictions.csv", index=False)
    test_df.to_csv(outdir / "test_predictions.csv", index=False)

    print("Baseline training complete.")
    print(f"Feature table: {(outdir / 'feature_table.parquet').resolve()}")
    print(f"Evaluation summary: {(outdir / 'evaluation_summary.json').resolve()}")
    print(json.dumps({"threshold": best_threshold, "test_f1": metrics['test']['f1']}, indent=2))


if __name__ == "__main__":
    main()
