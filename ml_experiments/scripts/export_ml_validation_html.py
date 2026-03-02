#!/usr/bin/env python3
"""Export a standalone HTML report for ML LOEO validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ML validation HTML from LOEO outputs.")
    p.add_argument(
        "--run-dir",
        default="outputs/ml/loeo_real_controls_v1",
        help="Directory containing loeo_aggregate_summary_all_bands.csv and loeo_event_summary_all_bands.csv",
    )
    p.add_argument("--out", default="outputs/ml_validation_dashboard.html")
    return p.parse_args()


def fig_div(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"responsive": True},
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg_path = run_dir / "loeo_aggregate_summary_all_bands.csv"
    evt_path = run_dir / "loeo_event_summary_all_bands.csv"
    bench_path = run_dir / "frozen_benchmark_report.json"

    if not agg_path.exists() or not evt_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {run_dir}: "
            "loeo_aggregate_summary_all_bands.csv and/or loeo_event_summary_all_bands.csv"
        )

    agg = pd.read_csv(agg_path)
    evt = pd.read_csv(evt_path)
    for c in ["policy_f1", "policy_far_per_year", "policy_precision", "policy_recall", "policy_lead_days"]:
        if c in evt.columns:
            evt[c] = pd.to_numeric(evt[c], errors="coerce")

    bench = {}
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))

    f1_fig = go.Figure()
    f1_fig.add_trace(
        go.Bar(
            x=agg["band"],
            y=agg["policy_f1_mean"],
            marker_color="#1f77b4",
            name="Mean F1",
        )
    )
    f1_fig.update_layout(
        title="Band-Level Mean F1",
        xaxis_title="Band",
        yaxis_title="F1",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    far_fig = go.Figure()
    far_fig.add_trace(
        go.Bar(
            x=agg["band"],
            y=agg["policy_far_mean"],
            marker_color="#d62728",
            name="Mean FAR/year",
        )
    )
    far_fig.add_hline(y=3.0, line_dash="dash", line_color="black")
    far_fig.update_layout(
        title="Band-Level Mean False Alarms / Year",
        xaxis_title="Band",
        yaxis_title="FAR/year",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    scatter = go.Figure()
    for band in sorted(evt["band"].dropna().unique()):
        b = evt[evt["band"] == band]
        scatter.add_trace(
            go.Scatter(
                x=b["policy_far_per_year"],
                y=b["policy_f1"],
                mode="markers+text",
                text=b["holdout_event"],
                textposition="top center",
                name=str(band),
            )
        )
    scatter.add_vline(x=3.0, line_dash="dash", line_color="black")
    scatter.update_layout(
        title="Event-Level Tradeoff: FAR vs F1",
        xaxis_title="False alarms per year",
        yaxis_title="F1",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )

    divs = [fig_div(f1_fig, include_js=True), fig_div(far_fig), fig_div(scatter)]

    bench_cards = ""
    if bench and bench.get("bands"):
        cards = []
        for b in bench["bands"]:
            cards.append(
                f"""
<div class="kpi">
  <div class="label">{b['band']} Benchmark F1</div>
  <div class="value">{float(b['benchmark_policy_f1']):.3f}</div>
  <div class="mini">FAR/year: {float(b['benchmark_policy_far_per_year']):.3f}</div>
  <div class="mini">Lead days: {float(b['benchmark_policy_lead_days']):.1f}</div>
  <div class="mini">First alert: {b['benchmark_first_alert_date']}</div>
</div>
"""
            )
        bench_cards = "\n".join(cards)

    table_cols = [
        c
        for c in [
            "band",
            "holdout_event",
            "policy_precision",
            "policy_recall",
            "policy_f1",
            "policy_far_per_year",
            "policy_first_alert_date",
            "policy_lead_days",
        ]
        if c in evt.columns
    ]
    table_html = evt[table_cols].sort_values(["band", "holdout_event"]).to_html(index=False, classes="tbl")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML Validation Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #102038; }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 18px; }}
    .header {{ background: #ffffff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ background: #fff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 8px; margin-bottom: 12px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 10px; margin-top: 10px; }}
    .kpi {{ background: #eff4fb; border: 1px solid #d8e0eb; border-radius: 10px; padding: 10px; }}
    .label {{ font-size: 12px; color: #37506d; }}
    .value {{ font-weight: 700; font-size: 20px; margin-top: 3px; }}
    .mini {{ font-size: 12px; color: #334e68; margin-top: 2px; }}
    .tbl {{ border-collapse: collapse; width: 100%; background: #fff; }}
    .tbl th, .tbl td {{ border: 1px solid #d8e0eb; padding: 6px 8px; font-size: 12px; }}
    .tbl th {{ background: #eef3fb; }}
    @media (max-width: 960px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h2 style="margin:0 0 6px 0;">ML Validation Dashboard (Cross-Event LOEO)</h2>
      <div>Run directory: <code>{run_dir.resolve()}</code></div>
      <div>Goal: train on other sinkholes and evaluate transfer performance, including frozen Eisenhower benchmark.</div>
      {"<div class='kpis'>" + bench_cards + "</div>" if bench_cards else ""}
    </div>
    <div class="grid2">
      <div class="panel">{divs[0]}</div>
      <div class="panel">{divs[1]}</div>
    </div>
    <div class="panel">{divs[2]}</div>
    <div class="panel">
      <h3 style="margin:6px 6px 10px 6px;">Event Summary</h3>
      {table_html}
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
