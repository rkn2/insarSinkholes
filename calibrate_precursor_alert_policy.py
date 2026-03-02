#!/usr/bin/env python3
"""Calibrate post-model alert policy to reduce false alarms.

Enhancements:
- Selection mode: val-only or event-wise cross-validation over train+val events.
- Additional policy gates: min persistence days + cooldown days.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate alert policy from prediction tables.")
    p.add_argument("--model-outdir", default="outputs/ml/night1_baseline")
    p.add_argument("--selection-mode", choices=["val_only", "event_cv"], default="event_cv")
    p.add_argument("--max-false-alarms-per-year", type=float, default=3.0)

    p.add_argument("--min-threshold", type=float, default=0.30)
    p.add_argument("--max-threshold", type=float, default=0.90)
    p.add_argument("--threshold-steps", type=int, default=31)

    p.add_argument("--max-consecutive", type=int, default=5)

    p.add_argument("--max-persistence-days", type=int, default=21)
    p.add_argument("--persistence-step-days", type=int, default=7)

    p.add_argument("--max-cooldown-days", type=int, default=30)
    p.add_argument("--cooldown-step-days", type=int, default=10)
    return p.parse_args()


def _apply_policy(
    df: pd.DataFrame,
    threshold: float,
    consecutive: int,
    min_persistence_days: int,
    cooldown_days: int,
) -> pd.DataFrame:
    out = df.copy().sort_values(["event_id", "date"]).reset_index(drop=True)
    out["raw_pred"] = (out["risk_prob"] >= threshold).astype(int)

    policy_pred = []
    for _, g in out.groupby("event_id", sort=False):
        raw = g["raw_pred"].to_numpy(dtype=int)
        dates = pd.to_datetime(g["date"], utc=True)

        run = 0
        run_start = None
        cooldown_until = None

        for i, v in enumerate(raw):
            t = dates.iloc[i]
            if v == 1:
                if run == 0:
                    run_start = t
                run += 1
            else:
                run = 0
                run_start = None

            enough_consecutive = run >= consecutive
            enough_persistence = False
            if run_start is not None:
                enough_persistence = (t - run_start).total_seconds() / 86400.0 >= float(min_persistence_days)

            cooldown_ok = cooldown_until is None or t >= cooldown_until
            is_alert = bool(enough_consecutive and enough_persistence and cooldown_ok)

            if is_alert:
                cooldown_until = t + pd.Timedelta(days=int(cooldown_days))
            policy_pred.append(1 if is_alert else 0)

    out["policy_pred"] = policy_pred
    return out


def _metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    y = df["risk_label"].to_numpy(dtype=int)
    p = df["policy_pred"].to_numpy(dtype=int)

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    pre = df[df["is_pre_event_window"]].copy()
    non_target = pre[pre["risk_label"] == 0]
    years = max(1e-6, len(non_target) / 52.0)
    false_alarms_per_year = float((non_target["policy_pred"] == 1).sum()) / years

    lead_days = []
    for _, g in df.groupby("event_id"):
        g = g.sort_values("date")
        alerts = g[(g["policy_pred"] == 1) & (g["is_pre_event_window"])].copy()
        if alerts.empty:
            continue
        first_alert = pd.to_datetime(alerts["date"].iloc[0], utc=True)
        event_date = pd.to_datetime(g["event_date"].iloc[0], utc=True)
        lead_days.append(float((event_date - first_alert).total_seconds() / 86400.0))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarms_per_year": float(false_alarms_per_year),
        "events_with_alert": int(len(lead_days)),
        "median_lead_days": float(np.median(lead_days)) if lead_days else None,
    }


def _event_cv_score(df: pd.DataFrame, policy: dict) -> dict:
    by_event = []
    for event_id, g in df.groupby("event_id"):
        applied = _apply_policy(
            g,
            threshold=policy["threshold"],
            consecutive=policy["consecutive"],
            min_persistence_days=policy["min_persistence_days"],
            cooldown_days=policy["cooldown_days"],
        )
        m = _metrics(applied)
        m["event_id"] = event_id
        by_event.append(m)

    edf = pd.DataFrame(by_event)
    return {
        "mean_f1": float(edf["f1"].mean()),
        "mean_recall": float(edf["recall"].mean()),
        "mean_precision": float(edf["precision"].mean()),
        "mean_false_alarms_per_year": float(edf["false_alarms_per_year"].mean()),
        "min_recall": float(edf["recall"].min()),
        "std_f1": float(edf["f1"].std(ddof=0)),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.model_outdir)

    val = pd.read_csv(outdir / "val_predictions.csv")
    test = pd.read_csv(outdir / "test_predictions.csv")
    train_path = outdir / "train_predictions.csv"
    train = pd.read_csv(train_path) if train_path.exists() else None

    for df in [d for d in [train, val, test] if d is not None]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
        df["is_pre_event_window"] = df["is_pre_event_window"].astype(bool)

    calib_df = val if args.selection_mode == "val_only" else pd.concat([train, val], ignore_index=True)

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.threshold_steps)
    persistence_values = list(range(0, args.max_persistence_days + 1, max(1, args.persistence_step_days)))
    cooldown_values = list(range(0, args.max_cooldown_days + 1, max(1, args.cooldown_step_days)))

    rows = []
    for th in thresholds:
        for cons in range(1, args.max_consecutive + 1):
            for persist in persistence_values:
                for cooldown in cooldown_values:
                    policy = {
                        "threshold": float(th),
                        "consecutive": int(cons),
                        "min_persistence_days": int(persist),
                        "cooldown_days": int(cooldown),
                    }

                    if args.selection_mode == "val_only":
                        c_applied = _apply_policy(calib_df, **policy)
                        c_metrics = _metrics(c_applied)
                        row = {
                            **policy,
                            "selector_score_f1": c_metrics["f1"],
                            "selector_score_recall": c_metrics["recall"],
                            "selector_score_precision": c_metrics["precision"],
                            "selector_false_alarms_per_year": c_metrics["false_alarms_per_year"],
                            "selector_min_recall": c_metrics["recall"],
                            "selector_std_f1": 0.0,
                        }
                    else:
                        cv = _event_cv_score(calib_df, policy)
                        row = {
                            **policy,
                            "selector_score_f1": cv["mean_f1"],
                            "selector_score_recall": cv["mean_recall"],
                            "selector_score_precision": cv["mean_precision"],
                            "selector_false_alarms_per_year": cv["mean_false_alarms_per_year"],
                            "selector_min_recall": cv["min_recall"],
                            "selector_std_f1": cv["std_f1"],
                        }

                    t_applied = _apply_policy(test, **policy)
                    t_metrics = _metrics(t_applied)
                    row.update(
                        {
                            "test_f1": t_metrics["f1"],
                            "test_recall": t_metrics["recall"],
                            "test_precision": t_metrics["precision"],
                            "test_false_alarms_per_year": t_metrics["false_alarms_per_year"],
                            "test_median_lead_days": t_metrics["median_lead_days"],
                            "test_events_with_alert": t_metrics["events_with_alert"],
                        }
                    )
                    rows.append(row)

    sweep = pd.DataFrame(rows).sort_values(
        ["threshold", "consecutive", "min_persistence_days", "cooldown_days"]
    ).reset_index(drop=True)

    feasible = sweep[sweep["selector_false_alarms_per_year"] <= args.max_false_alarms_per_year].copy()
    if feasible.empty:
        chosen = sweep.sort_values(["selector_score_f1", "selector_score_recall"], ascending=False).iloc[0]
        selection = "fallback_no_feasible_far_constraint"
    else:
        chosen = feasible.sort_values(
            ["selector_score_f1", "selector_min_recall", "selector_score_precision"],
            ascending=False,
        ).iloc[0]
        selection = "best_selector_f1_under_far_constraint"

    chosen_policy = {
        "threshold": float(chosen["threshold"]),
        "consecutive": int(chosen["consecutive"]),
        "min_persistence_days": int(chosen["min_persistence_days"]),
        "cooldown_days": int(chosen["cooldown_days"]),
    }

    chosen_calib = _apply_policy(calib_df, **chosen_policy)
    chosen_test = _apply_policy(test, **chosen_policy)

    calib_metrics = _metrics(chosen_calib)
    test_metrics = _metrics(chosen_test)

    report = {
        "selection_mode": args.selection_mode,
        "selection_rule": selection,
        "max_false_alarms_per_year": float(args.max_false_alarms_per_year),
        "chosen_policy": chosen_policy,
        "selector_metrics": calib_metrics,
        "test_metrics": test_metrics,
        "sweep_rows": int(len(sweep)),
    }

    sweep.to_csv(outdir / "alert_policy_sweep.csv", index=False)
    (outdir / "alert_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    chosen_calib.to_csv(outdir / "calibration_policy_predictions.csv", index=False)
    chosen_test.to_csv(outdir / "test_policy_predictions.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"Sweep CSV: {(outdir / 'alert_policy_sweep.csv').resolve()}")


if __name__ == "__main__":
    main()
