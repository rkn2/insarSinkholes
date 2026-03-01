#!/usr/bin/env python3
"""Export a standalone HTML version of the synthetic settlement dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.io as pio

import settlement_dashboard as sd


def fig_div(fig, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"responsive": True},
    )


def main() -> None:
    out_path = Path("outputs/settlement_dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b = sd.Building()
    df = sd.synthetic_timeseries(seed=7)
    cloud = sd.make_point_cloud(b)
    sensors = sd.accel_sensors(b)
    ps = sd.insar_ps_points(b)

    idx = len(df) - 1
    sinkhole_x_m, sinkhole_y_m, sigma_m, scale = 24.0, 12.0, 8.0, 1.0

    fused_mm = float(df.loc[idx, "fused_mm"]) * scale
    nearest_insar_mm = sd.nearest_insar_value(df["insar_mm"], idx) * scale
    settlement_mm = sd.settlement_field_mm(cloud, fused_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
    ps_settlement = sd.settlement_field_mm(
        ps.assign(surface=0), nearest_insar_mm, sinkhole_x_m, sinkhole_y_m, sigma_m
    )
    ps_settlement += np.random.default_rng(123).normal(0.0, 0.6, size=len(ps_settlement))

    figs = [
        sd.plan_view_fig(b, sinkhole_x_m, sinkhole_y_m, sigma_m, fused_mm, sensors),
        sd.section_view_fig(b, sinkhole_x_m, sigma_m, fused_mm),
        sd.point_cloud_fig(cloud, settlement_mm, ps, ps_settlement, sensors),
        sd.time_series_fig(df, idx),
        sd.modal_fig(df, idx),
    ]
    retro_df, retro_summary = sd.retrospective_data()
    retro_figs = []
    retro_html = ""
    if retro_df is not None and len(retro_df) > 0:
        ridx = len(retro_df) - 1
        retro_figs = [
            sd.insar_only_timeseries_fig(retro_df, ridx),
            sd.risk_score_fig(retro_df, ridx, retro_summary),
        ]
        retro_lead = retro_summary.get("lead_days_to_event")
        retro_break = retro_summary.get("slope_break_date")
        retro_p = retro_summary.get("slope_break_p_value")
        retro_html = f"""
    <div class="header">
      <h2 style="margin:0 0 6px 0;">Eisenhower Retrospective (InSAR-only)</h2>
      <div>Static snapshot from retrospective outputs with upgraded risk features.</div>
      <div class="kpis">
        <div class="kpi"><div class="label">Event Date</div><div class="value">{retro_summary.get("event_date", "n/a")}</div></div>
        <div class="kpi"><div class="label">First Alert</div><div class="value">{retro_summary.get("first_alert_date_in_claim_window", "n/a")}</div></div>
        <div class="kpi"><div class="label">Lead Days</div><div class="value">{retro_lead if retro_lead is not None else "n/a"}</div></div>
        <div class="kpi"><div class="label">Slope Break</div><div class="value">{retro_break if retro_break else "n/a"}</div></div>
      </div>
      <div style="margin-top:8px;color:#35526f;font-size:13px;">Slope-break p-value: {retro_p if retro_p is not None else "n/a"}</div>
    </div>
"""

    divs = [fig_div(figs[0], include_js=True)] + [fig_div(f, include_js=False) for f in figs[1:]]
    retro_divs = [fig_div(f, include_js=False) for f in retro_figs]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Settlement Structural Twin Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #102038; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    .header {{ background: #ffffff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin-top: 10px; }}
    .kpi {{ background: #eff4fb; border: 1px solid #d8e0eb; border-radius: 10px; padding: 10px; }}
    .kpi .label {{ font-size: 12px; color: #37506d; }}
    .kpi .value {{ font-weight: 700; font-size: 20px; margin-top: 3px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ background: #fff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 8px; margin-bottom: 12px; }}
    .note {{ background: #fffdf3; border: 1px solid #f0dfaa; border-radius: 12px; padding: 12px; }}
    @media (max-width: 960px) {{
      .grid2 {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h2 style="margin:0 0 6px 0;">Settlement-Focused Structural Digital Twin (Synthetic)</h2>
      <div>Synthetic fusion of InSAR and accelerometer indicators for sinkhole-sensitive monitoring.</div>
      <div class="kpis">
        <div class="kpi"><div class="label">Selected Date</div><div class="value">{df.loc[idx, "date"].date()}</div></div>
        <div class="kpi"><div class="label">Fused Settlement</div><div class="value">{fused_mm:.2f} mm</div></div>
        <div class="kpi"><div class="label">Nearest InSAR</div><div class="value">{nearest_insar_mm:.2f} mm</div></div>
        <div class="kpi"><div class="label">Modal Frequency</div><div class="value">{df.loc[idx, "modal_freq_hz"]:.3f} Hz</div></div>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">{divs[0]}</div>
      <div class="panel">{divs[1]}</div>
    </div>
    <div class="panel">{divs[2]}</div>
    <div class="grid2">
      <div class="panel">{divs[3]}</div>
      <div class="panel">{divs[4]}</div>
    </div>
    <div class="note">
      <b>Interpretation:</b> InSAR provides sparse absolute displacement anchors, accelerometers provide dense dynamic indicators, and the point-cloud projection maps likely settlement bowls to building geometry for sinkhole-risk triage.
    </div>
    {retro_html}
    {"<div class='grid2'><div class='panel'>" + retro_divs[0] + "</div><div class='panel'>" + retro_divs[1] + "</div></div>" if len(retro_divs) == 2 else ""}
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
