#!/usr/bin/env python3
"""Run hardened LOEO evaluation with controls, lead-time bands, and quality gating."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardened LOEO evaluation runner.")
    p.add_argument(
        "--feature-table",
        default="outputs/ml/derived_full_observed_aria_pairs_relaxed_w8/feature_table.parquet",
    )
    p.add_argument("--outdir", default="outputs/ml/loeo_hardened_v1")
    p.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    p.add_argument("--positive-class-weight", type=float, default=8.0)
    p.add_argument("--val-far-cap", type=float, default=3.0)
    p.add_argument("--exclude-features", nargs="*", default=["days_to_event", "robust_vel_z", "robust_accel_z"])
    p.add_argument("--max-false-alarms-per-year", type=float, default=3.0)
    p.add_argument("--threshold-steps", type=int, default=11)
    p.add_argument("--max-consecutive", type=int, default=3)
    p.add_argument("--max-persistence-days", type=int, default=14)
    p.add_argument("--persistence-step-days", type=int, default=7)
    p.add_argument("--max-cooldown-days", type=int, default=10)
    p.add_argument("--cooldown-step-days", type=int, default=10)
    p.add_argument("--quality-min-score", type=float, default=0.45)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument(
        "--real-controls-feature-csv",
        default="config/real_controls_features.csv",
        help="Real control feature rows to append (must include event_id,date,dataset,disp_mm,event_date).",
    )
    p.add_argument(
        "--bands",
        nargs="*",
        default=["30:90", "90:180"],
        help="Lead-time bands as min:max day ranges (inclusive).",
    )
    p.add_argument(
        "--frozen-benchmark-event",
        default="2023-08-16_eisenhower_parking_deck_penn_state",
        help="Event to report as frozen benchmark each run.",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )


def _parse_bands(text_bands: list[str]) -> list[tuple[str, int, int]]:
    out = []
    for b in text_bands:
        parts = b.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid band spec '{b}'. Use min:max, e.g. 30:90")
        lo = int(parts[0])
        hi = int(parts[1])
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError(f"Invalid band bounds in '{b}'")
        out.append((f"band_{lo}_{hi}", lo, hi))
    return out


def _add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["disp_mm"] = pd.to_numeric(out["disp_mm"], errors="coerce")
    out["acc_mm_day2"] = pd.to_numeric(out["acc_mm_day2"], errors="coerce")
    out["obs_gap_days"] = pd.to_numeric(out["obs_gap_days"], errors="coerce")

    out["obs_is_observed"] = (out.get("disp_source", pd.Series("", index=out.index)).astype(str) == "observed").astype(float)
    out["coherence_proxy"] = 0.0
    out["point_count_proxy"] = 0.0
    out["uncertainty_spread_mm"] = 0.0

    groups = []
    for _, g in out.sort_values(["event_id", "dataset", "date"]).groupby(["event_id", "dataset"], sort=False):
        gg = g.copy()
        acc_scale = float(np.nanquantile(np.abs(gg["acc_mm_day2"].to_numpy(dtype=float)), 0.9)) + 1e-6
        coh = np.exp(-np.abs(gg["acc_mm_day2"].to_numpy(dtype=float)) / acc_scale)
        gg["coherence_proxy"] = np.clip(coh, 0.0, 1.0)

        obs_frac = gg["obs_is_observed"].rolling(window=6, min_periods=1).mean().to_numpy(dtype=float)
        density = (6.0 / np.clip(gg["obs_gap_days"].fillna(6.0).to_numpy(dtype=float), 1.0, 60.0))
        density = np.clip(density, 0.0, 1.0)
        gg["point_count_proxy"] = np.clip(20.0 + 80.0 * (0.7 * obs_frac + 0.3 * density), 0.0, 100.0)

        spread = gg["disp_mm"].rolling(window=6, min_periods=2).std().fillna(0.0).to_numpy(dtype=float)
        gg["uncertainty_spread_mm"] = np.clip(spread, 0.0, None)
        groups.append(gg)

    out = pd.concat(groups, ignore_index=True)
    spread_scale = float(np.nanquantile(out["uncertainty_spread_mm"].to_numpy(dtype=float), 0.9)) + 1e-6
    spread_quality = 1.0 / (1.0 + (out["uncertainty_spread_mm"] / spread_scale))
    point_quality = np.clip(out["point_count_proxy"] / 100.0, 0.0, 1.0)
    out["quality_score"] = np.clip(
        0.45 * out["coherence_proxy"] + 0.35 * point_quality + 0.20 * spread_quality,
        0.0,
        1.0,
    )
    out = out.drop(columns=["obs_is_observed"])
    return out


def _load_real_controls(path: str, base_cols: list[str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Real controls file not found: {p}")
    ctrl = pd.read_csv(p)
    required = {"event_id", "dataset", "date", "disp_mm", "event_date", "source_event_id"}
    missing = required - set(ctrl.columns)
    if missing:
        raise RuntimeError(f"Real controls file missing columns: {sorted(missing)}")

    ctrl["date"] = pd.to_datetime(ctrl["date"], errors="coerce", utc=True)
    ctrl["event_date"] = pd.to_datetime(ctrl["event_date"], errors="coerce", utc=True)
    ctrl["disp_mm"] = pd.to_numeric(ctrl["disp_mm"], errors="coerce")
    ctrl = ctrl.dropna(subset=["date", "event_date", "disp_mm"]).copy()
    if ctrl.empty:
        raise RuntimeError("Real controls file has no valid rows after parsing.")

    # Fill any absent feature columns with defaults; quality features are rebuilt later.
    for c in base_cols:
        if c not in ctrl.columns:
            if c in {"split"}:
                ctrl[c] = "train"
            elif c in {"risk_label", "is_pre_event_window", "changepoint_flag", "is_aria", "is_opera"}:
                ctrl[c] = 0
            else:
                ctrl[c] = 0.0

    ctrl["days_since_start"] = (
        ctrl.groupby(["event_id", "dataset"])["date"].transform(lambda s: (s - s.min()).dt.total_seconds() / 86400.0)
    )
    ctrl["days_to_event"] = (ctrl["event_date"] - ctrl["date"]).dt.total_seconds() / 86400.0
    ctrl["is_pre_event_window"] = (ctrl["days_to_event"] >= 0.0).astype(bool)
    ctrl["split"] = "train"
    ctrl["risk_label"] = 0
    ctrl["dataset"] = ctrl["dataset"].astype(str)
    ctrl["is_aria"] = (ctrl["dataset"] == "ARIA_S1_GUNW").astype(int)
    ctrl["is_opera"] = (ctrl["dataset"] == "OPERA_S1_DISP").astype(int)

    return ctrl[sorted(set(base_cols + ["source_event_id"]))]


def _false_alarms_per_year(df: pd.DataFrame, pred_col: str) -> float:
    pre = df[df["is_pre_event_window"].astype(bool)].copy()
    non_target = pre[pre["risk_label"] == 0]
    if non_target.empty:
        return 0.0
    years = max(1e-6, len(non_target) / 52.0)
    return float((non_target[pred_col].astype(int) == 1).sum()) / years


def _event_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float | int | str | None]:
    y = df["risk_label"].astype(int).to_numpy()
    p = df[pred_col].astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    pre_alerts = (
        df[(df[pred_col].astype(int) == 1) & (df["is_pre_event_window"].astype(bool))]
        .sort_values("date")
        .reset_index(drop=True)
    )
    first_alert_date = None
    lead_days = None
    if not pre_alerts.empty:
        first_alert_ts = pd.to_datetime(pre_alerts.loc[0, "date"], utc=True)
        event_ts = pd.to_datetime(df["event_date"].iloc[0], utc=True)
        first_alert_date = first_alert_ts.date().isoformat()
        lead_days = float((event_ts - first_alert_ts).total_seconds() / 86400.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarms_per_year": float(_false_alarms_per_year(df, pred_col)),
        "first_alert_date": first_alert_date,
        "lead_days": lead_days,
        "n_rows": int(len(df)),
        "n_positive_labels": int(y.sum()),
    }


def _run_band(
    feat: pd.DataFrame,
    sinkhole_events: list[str],
    band_name: str,
    band_min: int,
    band_max: int,
    args: argparse.Namespace,
    outdir: Path,
) -> pd.DataFrame:
    band_dir = outdir / band_name
    folds_dir = band_dir / "folds"
    tmp_dir = band_dir / "tmp"
    folds_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, holdout_event in enumerate(sinkhole_events):
        remaining = [e for e in sinkhole_events if e != holdout_event]
        val_event = remaining[i % len(remaining)]
        train_events = [e for e in remaining if e != val_event]

        fold_name = f"{i+1:02d}_{holdout_event}"
        fold_dir = folds_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_feat = feat.copy()
        fold_feat["split"] = "train"
        fold_feat.loc[fold_feat["event_id"] == val_event, "split"] = "val"
        fold_feat.loc[fold_feat["event_id"] == holdout_event, "split"] = "test"

        # Controls inherit split from their source sinkhole event.
        ctrl_mask = fold_feat["event_id"].astype(str).str.startswith("control__")
        if "source_event_id" in fold_feat.columns:
            ctrl_mask = ctrl_mask | fold_feat["source_event_id"].notna()
        if ctrl_mask.any():
            if "source_event_id" in fold_feat.columns:
                src = (
                    fold_feat.loc[ctrl_mask, "source_event_id"]
                    .fillna(fold_feat.loc[ctrl_mask, "event_id"].astype(str).str.replace("control__", "", regex=False))
                    .astype(str)
                )
            else:
                src = fold_feat.loc[ctrl_mask, "event_id"].astype(str).str.replace("control__", "", regex=False)
            src_split = src.map(
                {
                    holdout_event: "test",
                    val_event: "val",
                }
            ).fillna("train")
            fold_feat.loc[ctrl_mask, "split"] = src_split.to_numpy()

        fold_feat_path = tmp_dir / f"{fold_name}_feature_table.parquet"
        fold_feat.to_parquet(fold_feat_path, index=False)

        clf_out = fold_dir / "classifier"
        cmd_train = [
            sys.executable,
            "train_precursor_classifier.py",
            "--feature-table",
            str(fold_feat_path),
            "--outdir",
            str(clf_out),
            "--model",
            args.model,
            "--positive-class-weight",
            str(args.positive_class_weight),
            "--val-far-cap",
            str(args.val_far_cap),
            "--relabel-min-days",
            str(band_min),
            "--relabel-max-days",
            str(band_max),
            "--exclude-features",
            *args.exclude_features,
        ]
        _run(cmd_train)

        cmd_cal = [
            sys.executable,
            "calibrate_precursor_alert_policy.py",
            "--model-outdir",
            str(clf_out),
            "--selection-mode",
            "event_cv",
            "--max-false-alarms-per-year",
            str(args.max_false_alarms_per_year),
            "--threshold-steps",
            str(args.threshold_steps),
            "--max-consecutive",
            str(args.max_consecutive),
            "--max-persistence-days",
            str(args.max_persistence_days),
            "--persistence-step-days",
            str(args.persistence_step_days),
            "--max-cooldown-days",
            str(args.max_cooldown_days),
            "--cooldown-step-days",
            str(args.cooldown_step_days),
            "--quality-score-col",
            "quality_score",
            "--quality-min-score",
            str(args.quality_min_score),
        ]
        _run(cmd_cal)

        raw_test = pd.read_csv(clf_out / "test_predictions.csv")
        pol_test = pd.read_csv(clf_out / "test_policy_predictions.csv")
        raw_test["date"] = pd.to_datetime(raw_test["date"], errors="coerce", utc=True)
        pol_test["date"] = pd.to_datetime(pol_test["date"], errors="coerce", utc=True)
        raw_test["event_date"] = pd.to_datetime(raw_test["event_date"], errors="coerce", utc=True)
        pol_test["event_date"] = pd.to_datetime(pol_test["event_date"], errors="coerce", utc=True)

        raw_ev = raw_test[raw_test["event_id"] == holdout_event].copy()
        pol_ev = pol_test[pol_test["event_id"] == holdout_event].copy()
        if raw_ev.empty or pol_ev.empty:
            raise RuntimeError(f"Missing holdout rows for {holdout_event}")

        with open(clf_out / "evaluation_summary.json", "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
        with open(clf_out / "alert_policy_report.json", "r", encoding="utf-8") as f:
            policy_report = json.load(f)

        raw_m = _event_metrics(raw_ev, "risk_pred")
        pol_m = _event_metrics(pol_ev, "policy_pred")
        row = {
            "band": band_name,
            "band_min_days": band_min,
            "band_max_days": band_max,
            "fold": fold_name,
            "holdout_event": holdout_event,
            "val_event": val_event,
            "train_events": "|".join(train_events),
            "classifier_threshold_on_val": float(eval_summary.get("threshold_tuned_on_val", np.nan)),
            "policy_threshold": float(policy_report["chosen_policy"]["threshold"]),
            "policy_consecutive": int(policy_report["chosen_policy"]["consecutive"]),
            "policy_persistence_days": int(policy_report["chosen_policy"]["min_persistence_days"]),
            "policy_cooldown_days": int(policy_report["chosen_policy"]["cooldown_days"]),
            "quality_min_score": float(args.quality_min_score),
            "raw_precision": raw_m["precision"],
            "raw_recall": raw_m["recall"],
            "raw_f1": raw_m["f1"],
            "raw_far_per_year": raw_m["false_alarms_per_year"],
            "raw_first_alert_date": raw_m["first_alert_date"],
            "raw_lead_days": raw_m["lead_days"],
            "policy_precision": pol_m["precision"],
            "policy_recall": pol_m["recall"],
            "policy_f1": pol_m["f1"],
            "policy_far_per_year": pol_m["false_alarms_per_year"],
            "policy_first_alert_date": pol_m["first_alert_date"],
            "policy_lead_days": pol_m["lead_days"],
            "n_rows": int(raw_m["n_rows"]),
            "n_positive_labels": int(raw_m["n_positive_labels"]),
        }
        rows.append(row)
        print(json.dumps({"band": band_name, "completed_fold": fold_name, "holdout_event": holdout_event}, indent=2))

    summary = pd.DataFrame(rows).sort_values(["holdout_event"]).reset_index(drop=True)
    summary.to_csv(band_dir / "loeo_event_summary.csv", index=False)
    agg = {
        "band": band_name,
        "band_min_days": int(band_min),
        "band_max_days": int(band_max),
        "n_folds": int(len(summary)),
        "policy_f1_mean": float(summary["policy_f1"].mean()),
        "policy_f1_median": float(summary["policy_f1"].median()),
        "policy_far_mean": float(summary["policy_far_per_year"].mean()),
        "policy_far_median": float(summary["policy_far_per_year"].median()),
        "policy_lead_days_mean": float(summary["policy_lead_days"].dropna().mean()),
        "policy_lead_days_median": float(summary["policy_lead_days"].dropna().median()),
        "events_with_policy_far_le_3": int((summary["policy_far_per_year"] <= 3.0).sum()),
    }
    (band_dir / "loeo_aggregate_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return summary


def _write_benchmark_report(
    all_summary: pd.DataFrame,
    frozen_event: str,
    outdir: Path,
) -> None:
    event_rows = all_summary[all_summary["holdout_event"] == frozen_event].copy()
    if event_rows.empty:
        raise RuntimeError(f"Frozen benchmark event '{frozen_event}' not found in LOEO results.")

    report = {
        "frozen_benchmark_event": frozen_event,
        "bands": [],
    }
    lines = ["# Frozen Benchmark Report", ""]
    lines.append(f"- Event: `{frozen_event}`")
    lines.append("")

    for band in sorted(all_summary["band"].unique()):
        band_df = all_summary[all_summary["band"] == band].copy()
        br = event_rows[event_rows["band"] == band].iloc[0]
        item = {
            "band": band,
            "benchmark_policy_f1": float(br["policy_f1"]),
            "benchmark_policy_far_per_year": float(br["policy_far_per_year"]),
            "benchmark_policy_lead_days": None if pd.isna(br["policy_lead_days"]) else float(br["policy_lead_days"]),
            "benchmark_first_alert_date": None if pd.isna(br["policy_first_alert_date"]) else str(br["policy_first_alert_date"]),
            "aggregate_policy_f1_median": float(band_df["policy_f1"].median()),
            "aggregate_policy_far_median": float(band_df["policy_far_per_year"].median()),
            "aggregate_events_with_far_le_3": int((band_df["policy_far_per_year"] <= 3.0).sum()),
            "aggregate_n_events": int(len(band_df)),
        }
        report["bands"].append(item)
        lines.append(f"## {band}")
        lines.append(f"- Benchmark F1: {item['benchmark_policy_f1']:.3f}")
        lines.append(f"- Benchmark FAR/year: {item['benchmark_policy_far_per_year']:.3f}")
        lines.append(f"- Benchmark lead days: {item['benchmark_policy_lead_days']}")
        lines.append(f"- Benchmark first alert: {item['benchmark_first_alert_date']}")
        lines.append(f"- LOEO median F1: {item['aggregate_policy_f1_median']:.3f}")
        lines.append(f"- LOEO median FAR/year: {item['aggregate_policy_far_median']:.3f}")
        lines.append(f"- Events FAR<=3: {item['aggregate_events_with_far_le_3']}/{item['aggregate_n_events']}")
        lines.append("")

    (outdir / "frozen_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "frozen_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bands = _parse_bands(args.bands)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    feat = pd.read_parquet(args.feature_table)
    if "event_id" not in feat.columns:
        raise ValueError("Feature table missing event_id")

    feat = _add_quality_features(feat)
    sinkhole_events = sorted(str(x) for x in feat["event_id"].unique() if not str(x).startswith("control__"))
    if len(sinkhole_events) < 3:
        raise RuntimeError("Need at least 3 sinkhole events.")

    if not args.no_controls:
        base_cols = list(feat.columns)
        ctrl = _load_real_controls(args.real_controls_feature_csv, base_cols=base_cols)
        feat["source_event_id"] = feat["event_id"].astype(str)
        for c in feat.columns:
            if c not in ctrl.columns:
                ctrl[c] = np.nan
        for c in ctrl.columns:
            if c not in feat.columns:
                feat[c] = np.nan
        feat = pd.concat([feat, ctrl[feat.columns]], ignore_index=True)
    feat["is_pre_event_window"] = feat["is_pre_event_window"].astype(bool)
    feat["risk_label"] = pd.to_numeric(feat["risk_label"], errors="coerce").fillna(0).astype(int)

    all_rows = []
    for band_name, lo, hi in bands:
        summary = _run_band(
            feat=feat,
            sinkhole_events=sinkhole_events,
            band_name=band_name,
            band_min=lo,
            band_max=hi,
            args=args,
            outdir=outdir,
        )
        all_rows.append(summary)

    full = pd.concat(all_rows, ignore_index=True).sort_values(["band", "holdout_event"]).reset_index(drop=True)
    full.to_csv(outdir / "loeo_event_summary_all_bands.csv", index=False)

    agg_rows = []
    for band in sorted(full["band"].unique()):
        b = full[full["band"] == band]
        agg_rows.append(
            {
                "band": band,
                "policy_f1_mean": float(b["policy_f1"].mean()),
                "policy_f1_median": float(b["policy_f1"].median()),
                "policy_far_mean": float(b["policy_far_per_year"].mean()),
                "policy_far_median": float(b["policy_far_per_year"].median()),
                "policy_lead_days_mean": float(b["policy_lead_days"].dropna().mean()),
                "policy_lead_days_median": float(b["policy_lead_days"].dropna().median()),
                "events_with_policy_far_le_3": int((b["policy_far_per_year"] <= 3.0).sum()),
                "n_events": int(len(b)),
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(outdir / "loeo_aggregate_summary_all_bands.csv", index=False)

    _write_benchmark_report(
        all_summary=full,
        frozen_event=args.frozen_benchmark_event,
        outdir=outdir,
    )

    print(
        json.dumps(
            {
                "outdir": str(outdir.resolve()),
                "bands": [b[0] for b in bands],
                "controls_enabled": not args.no_controls,
                "quality_min_score": args.quality_min_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
