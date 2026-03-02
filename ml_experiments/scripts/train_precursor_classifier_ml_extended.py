#!/usr/bin/env python3
"""Train an imbalanced classifier for sinkhole precursor risk.

Consumes an existing feature_table.parquet and produces calibrated probabilities
and split-wise prediction CSVs for downstream alert-policy calibration.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train calibrated classifier on feature table.")
    p.add_argument(
        "--feature-table",
        default="outputs/ml/derived_full_observed_aria_pairs_relaxed_w8/feature_table.parquet",
    )
    p.add_argument("--outdir", default="outputs/ml/classifier_v1")
    p.add_argument("--model", choices=["logreg", "hgb"], default="hgb")
    p.add_argument("--positive-class-weight", type=float, default=12.0)
    p.add_argument("--val-far-cap", type=float, default=3.0)
    p.add_argument(
        "--relabel-window-days",
        type=int,
        default=None,
        help="If set, rebuild risk_label as 1 when 0<=days_to_event<=window_days, else 0.",
    )
    p.add_argument(
        "--relabel-min-days",
        type=float,
        default=None,
        help="Lower bound for relabel band (inclusive). Used with --relabel-max-days.",
    )
    p.add_argument(
        "--relabel-max-days",
        type=float,
        default=None,
        help="Upper bound for relabel band (inclusive). Used with --relabel-min-days.",
    )
    p.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        help="Feature columns to exclude from model fitting.",
    )
    return p.parse_args()


def _false_alarms_per_year(df: pd.DataFrame, pred_col: str) -> float:
    pre = df[df["is_pre_event_window"]].copy()
    non_target = pre[pre["risk_label"] == 0]
    if non_target.empty:
        return 0.0
    years = max(1e-6, len(non_target) / 52.0)
    return float((non_target[pred_col] == 1).sum()) / years


def _metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y = df["risk_label"].to_numpy(dtype=int)
    p = df[pred_col].to_numpy(dtype=int)
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "false_alarms_per_year": float(_false_alarms_per_year(df, pred_col)),
        "n": int(len(df)),
        "positives": int(y.sum()),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(args.feature_table)
    req = {
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
        "event_date",
    }
    missing = req - set(feat.columns)
    if missing:
        raise ValueError(f"feature table missing required columns: {sorted(missing)}")

    feat["date"] = pd.to_datetime(feat["date"], errors="coerce", utc=True)
    feat["event_date"] = pd.to_datetime(feat["event_date"], errors="coerce", utc=True)

    pre = feat[feat["is_pre_event_window"]].copy().reset_index(drop=True)

    if args.relabel_window_days is not None:
        pre["risk_label"] = (
            (pre["days_to_event"] >= 0.0) & (pre["days_to_event"] <= float(args.relabel_window_days))
        ).astype(int)
    if args.relabel_min_days is not None or args.relabel_max_days is not None:
        if args.relabel_min_days is None or args.relabel_max_days is None:
            raise RuntimeError("Use both --relabel-min-days and --relabel-max-days together.")
        lo = float(args.relabel_min_days)
        hi = float(args.relabel_max_days)
        if lo > hi:
            raise RuntimeError("--relabel-min-days must be <= --relabel-max-days.")
        pre["risk_label"] = ((pre["days_to_event"] >= lo) & (pre["days_to_event"] <= hi)).astype(int)

    train_df = pre[pre["split"] == "train"].copy()
    val_df = pre[pre["split"] == "val"].copy()
    test_df = pre[pre["split"] == "test"].copy()
    if any(d.empty for d in [train_df, val_df, test_df]):
        raise RuntimeError("Train/val/test splits must all be non-empty.")

    feature_cols = [
        "disp_mm",
        "cum_settlement_mm",
        "vel_mm_day",
        "acc_mm_day2",
        "robust_vel_z",
        "robust_accel_z",
        "changepoint_flag",
        "obs_gap_days",
        "is_aria",
        "is_opera",
        "days_to_event",
    ]
    for opt_col in ["coherence_proxy", "point_count_proxy", "uncertainty_spread_mm", "quality_score"]:
        if opt_col in pre.columns and opt_col not in feature_cols:
            feature_cols.append(opt_col)
    if args.exclude_features:
        blocked = set(args.exclude_features)
        feature_cols = [c for c in feature_cols if c not in blocked]
    for c in feature_cols:
        if c not in pre.columns:
            raise RuntimeError(f"Missing model feature column: {c}")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["risk_label"].to_numpy(dtype=int)
    sw = np.where(y_train == 1, float(args.positive_class_weight), 1.0)

    if args.model == "logreg":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight={0: 1.0, 1: float(args.positive_class_weight)},
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        base.fit(X_train, y_train)
    else:
        base = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=0.3,
            random_state=42,
        )
        base.fit(X_train, y_train, sample_weight=sw)

    # Probability calibration on validation split to reduce miscalibration.
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["risk_label"].to_numpy(dtype=int)
    calibrator = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)

    def infer(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        X = out[feature_cols].to_numpy(dtype=float)
        out["risk_prob"] = calibrator.predict_proba(X)[:, 1]
        return out

    train_pred = infer(train_df)
    val_pred = infer(val_df)
    test_pred = infer(test_df)

    # threshold tune: maximize val F1 under FAR cap
    best_th = 0.5
    best_score = -1.0
    for th in np.linspace(0.1, 0.95, 35):
        v = val_pred.copy()
        v["risk_pred"] = (v["risk_prob"] >= th).astype(int)
        m = _metrics(v, "risk_pred")
        score = m["f1"] if m["false_alarms_per_year"] <= float(args.val_far_cap) else -1.0
        if score > best_score:
            best_score = score
            best_th = float(th)

    for d in [train_pred, val_pred, test_pred]:
        d["risk_pred"] = (d["risk_prob"] >= best_th).astype(int)

    eval_summary = {
        "model": args.model,
        "positive_class_weight": float(args.positive_class_weight),
        "calibration": "isotonic_on_val",
        "threshold_tuned_on_val": best_th,
        "val_far_cap": float(args.val_far_cap),
        "relabel_window_days": args.relabel_window_days,
        "relabel_min_days": args.relabel_min_days,
        "relabel_max_days": args.relabel_max_days,
        "excluded_features": list(args.exclude_features),
        "train": _metrics(train_pred, "risk_pred"),
        "val": _metrics(val_pred, "risk_pred"),
        "test": _metrics(test_pred, "risk_pred"),
        "feature_columns": feature_cols,
    }

    with open(outdir / "model.pkl", "wb") as f:
        pickle.dump({"base": base, "calibrator": calibrator, "features": feature_cols}, f)

    (outdir / "evaluation_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    train_pred.to_csv(outdir / "train_predictions.csv", index=False)
    val_pred.to_csv(outdir / "val_predictions.csv", index=False)
    test_pred.to_csv(outdir / "test_predictions.csv", index=False)

    print("Classifier training complete.")
    print(f"Output: {outdir.resolve()}")
    print(json.dumps({"threshold": best_th, "test": eval_summary["test"]}, indent=2))


if __name__ == "__main__":
    main()
