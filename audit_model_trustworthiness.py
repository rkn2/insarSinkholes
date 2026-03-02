#!/usr/bin/env python3
"""Trustworthiness audit for sinkhole precursor classifier outputs.

Checks for:
- Potential feature leakage into labels.
- Split contamination / duplicate leakage.
- Overly predictive single-feature shortcuts.
- Counterfactual performance after removing leakage-suspect features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit trustworthiness of classifier results.")
    p.add_argument("--model-outdir", default="outputs/ml/classifier_v1_hgb")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    return p.parse_args()


def _metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y = df["risk_label"].to_numpy(dtype=int)
    p = df[pred_col].to_numpy(dtype=int)
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "n": int(len(df)),
        "positives": int(y.sum()),
    }


def _best_threshold(val: pd.DataFrame) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, 37):
        pred = (val["risk_prob"].to_numpy() >= th).astype(int)
        f1 = f1_score(val["risk_label"].to_numpy(), pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(th)
    return best_t


def _single_feature_scan(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    ytr = train["risk_label"].to_numpy(dtype=int)
    yte = test["risk_label"].to_numpy(dtype=int)

    for c in cols:
        if c not in train.columns:
            continue
        xtr = pd.to_numeric(train[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
        xte = pd.to_numeric(test[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=7)
        clf.fit(xtr, ytr)
        proba_te = clf.predict_proba(xte)[:, 1]
        # tune threshold on train proxy
        proba_tr = clf.predict_proba(xtr)[:, 1]
        best_t = 0.5
        best_f1 = -1.0
        for th in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(ytr, (proba_tr >= th).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(th)
        pred_te = (proba_te >= best_t).astype(int)
        out.append(
            {
                "feature": c,
                "test_f1_single_feature": float(f1_score(yte, pred_te, zero_division=0)),
                "test_precision_single_feature": float(precision_score(yte, pred_te, zero_division=0)),
                "test_recall_single_feature": float(recall_score(yte, pred_te, zero_division=0)),
            }
        )
    return sorted(out, key=lambda r: r["test_f1_single_feature"], reverse=True)


def _leakage_ablation(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], remove: set[str]) -> dict:
    kept = [c for c in feature_cols if c not in remove and c in train.columns]
    Xtr = train[kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    Xv = val[kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    Xte = test[kept].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    ytr = train["risk_label"].to_numpy(dtype=int)
    yv = val["risk_label"].to_numpy(dtype=int)
    yte = test["risk_label"].to_numpy(dtype=int)

    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=400, random_state=11)
    clf.fit(Xtr, ytr)

    pv = clf.predict_proba(Xv)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]

    best_t = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(yv, (pv >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(th)

    pred_te = (pte >= best_t).astype(int)
    return {
        "kept_features": kept,
        "removed_features": sorted(remove),
        "test_f1": float(f1_score(yte, pred_te, zero_division=0)),
        "test_precision": float(precision_score(yte, pred_te, zero_division=0)),
        "test_recall": float(recall_score(yte, pred_te, zero_division=0)),
        "threshold": best_t,
    }


def main() -> None:
    args = parse_args()
    base = Path(args.model_outdir)

    train = pd.read_csv(base / "train_predictions.csv")
    val = pd.read_csv(base / "val_predictions.csv")
    test = pd.read_csv(base / "test_predictions.csv")

    for d in [train, val, test]:
        d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce", utc=True)

    full = pd.concat([train.assign(_split="train"), val.assign(_split="val"), test.assign(_split="test")], ignore_index=True)

    # Leakage risk flags based on label construction + unavailable-at-inference features.
    leakage_suspects = [
        "days_to_event",      # unavailable in real-time; needs known event date
        "robust_vel_z",       # currently used directly in label rule
        "robust_accel_z",     # currently used directly in label rule
    ]

    # Split contamination check via near-duplicate keys.
    key_cols = [c for c in ["event_id", "dataset", "date"] if c in full.columns]
    dup_cross_split = 0
    if key_cols:
        k = full[key_cols + ["_split"]].copy()
        g = k.groupby(key_cols)["_split"].nunique().reset_index(name="n_splits")
        dup_cross_split = int((g["n_splits"] > 1).sum())

    # Temporal sanity check: pre-event-only assumption.
    post_event_rows = int((full["date"] > full["event_date"]).sum())

    # Baseline metrics from saved predictions.
    base_metrics = {
        "train": _metrics(train, "risk_pred"),
        "val": _metrics(val, "risk_pred"),
        "test": _metrics(test, "risk_pred"),
    }

    # Single-feature scan for shortcut signals.
    candidate_features = [
        "days_to_event",
        "robust_vel_z",
        "robust_accel_z",
        "disp_mm",
        "vel_mm_day",
        "acc_mm_day2",
        "cum_settlement_mm",
    ]
    shortcut_scan = _single_feature_scan(train, test, candidate_features)

    # Counterfactual ablation: remove suspected leakage features.
    feature_cols = [
        c
        for c in [
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
        if c in train.columns
    ]

    ablation = _leakage_ablation(
        train,
        val,
        test,
        feature_cols=feature_cols,
        remove={"days_to_event", "robust_vel_z", "robust_accel_z"},
    )

    # Simple verdict logic.
    findings = []
    if "days_to_event" in train.columns:
        findings.append("`days_to_event` is inference-leaky for real-time deployment.")
    if dup_cross_split > 0:
        findings.append(f"Found {dup_cross_split} key collisions across splits (event_id,dataset,date).")
    if post_event_rows > 0:
        findings.append(f"Found {post_event_rows} rows after event_date in prediction tables.")

    top_shortcut = shortcut_scan[0] if shortcut_scan else None
    if top_shortcut and top_shortcut["test_f1_single_feature"] >= 0.8:
        findings.append(
            f"Single feature `{top_shortcut['feature']}` alone yields very high test F1 ({top_shortcut['test_f1_single_feature']:.3f}), indicating shortcut risk."
        )

    verdict = "high_leakage_risk" if findings else "no_major_leakage_flags"

    report = {
        "verdict": verdict,
        "findings": findings,
        "baseline_metrics": base_metrics,
        "split_contamination": {
            "cross_split_key_collisions": dup_cross_split,
            "key_columns": key_cols,
        },
        "temporal_sanity": {
            "post_event_rows_in_predictions": post_event_rows,
        },
        "shortcut_scan_top": shortcut_scan[:5],
        "counterfactual_ablation": ablation,
        "recommended_actions": [
            "Remove days_to_event from deployment model features.",
            "Do not define labels using robust_vel_z/robust_accel_z if those are model inputs.",
            "Rebuild labels from external event windows only, then recompute model features independently.",
            "Re-run event-wise CV after leakage feature removal and compare holdout stability.",
        ],
    }

    out_json = Path(args.out_json) if args.out_json else base / "trustworthiness_audit.json"
    out_md = Path(args.out_md) if args.out_md else base / "trustworthiness_audit.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = []
    md.append("# Trustworthiness Audit")
    md.append("")
    md.append(f"- Verdict: **{report['verdict']}**")
    md.append(f"- Findings count: **{len(findings)}**")
    md.append("")
    if findings:
        md.append("## Findings")
        for f in findings:
            md.append(f"- {f}")
        md.append("")
    md.append("## Baseline Test Metrics")
    bt = report["baseline_metrics"]["test"]
    md.append(f"- Precision: {bt['precision']:.4f}")
    md.append(f"- Recall: {bt['recall']:.4f}")
    md.append(f"- F1: {bt['f1']:.4f}")
    md.append("")
    md.append("## Counterfactual (Leakage Features Removed)")
    ab = report["counterfactual_ablation"]
    md.append(f"- Removed: {', '.join(ab['removed_features'])}")
    md.append(f"- Test F1: {ab['test_f1']:.4f}")
    md.append(f"- Test Precision: {ab['test_precision']:.4f}")
    md.append(f"- Test Recall: {ab['test_recall']:.4f}")
    md.append("")
    md.append("## Recommended Actions")
    for a in report["recommended_actions"]:
        md.append(f"- {a}")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Audit JSON: {out_json}")
    print(f"Audit MD: {out_md}")
    print(json.dumps({"verdict": verdict, "findings": findings, "test_f1": bt["f1"], "ablation_test_f1": ab["test_f1"]}, indent=2))


if __name__ == "__main__":
    main()
