#!/usr/bin/env python3
"""Export a standalone HTML snapshot focused on Eisenhower retrospective outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

import settlement_dashboard as sd


def fig_div(fig, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"responsive": True},
    )


def _retrospective_visual_state(retro_df: pd.DataFrame):
    real_geom = sd.load_eisenhower_geometry()
    if real_geom is not None:
        b = real_geom["building"]
        cloud = real_geom["cloud"]
        footprint_xy = real_geom["footprint"]
        real_ps_obs = sd.load_real_insar_points_local(real_geom["minx"], real_geom["miny"])
        sinkhole_xy = sd.load_sinkhole_location_local(real_geom["minx"], real_geom["miny"])
        if sinkhole_xy is None:
            sinkhole_xy = (float(real_geom["sinkhole"]["x"]), float(real_geom["sinkhole"]["y"]))
        sigma_m = float(real_geom["sinkhole"]["sigma"])
    else:
        b = sd.Building()
        cloud = sd.make_point_cloud(b)
        footprint_xy = None
        real_ps_obs = None
        sinkhole_xy = (24.0, 12.0)
        sigma_m = 8.0

    idx = len(retro_df) - 1
    fused_mm = float(retro_df.loc[idx, "smoothed_mm"]) if "smoothed_mm" in retro_df.columns else float(retro_df.loc[idx, "displacement_mm"])
    nearest_insar_mm = float(retro_df.loc[idx, "displacement_mm"])
    sinkhole_x_m, sinkhole_y_m = float(sinkhole_xy[0]), float(sinkhole_xy[1])

    settlement_mm = sd.settlement_field_mm(cloud, fused_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
    interp_grid = None
    section_profile = None
    if real_ps_obs is not None and len(real_ps_obs) > 0 and "point_id" in real_ps_obs.columns:
        curr_date = pd.Timestamp(retro_df.loc[idx, "date"])
        tmp = real_ps_obs.copy()
        tmp["d"] = (tmp["date"] - curr_date).abs()
        near = tmp.sort_values(["point_id", "d"]).groupby("point_id", as_index=False).first()
        ps = near[["x", "y"]].copy()
        ps["z"] = b.height_m
        ps_settlement = near["disp_mm"].to_numpy()
        settlement_mm = sd.idw_interpolate(
            ps["x"].to_numpy(),
            ps["y"].to_numpy(),
            ps_settlement,
            cloud["x"].to_numpy(),
            cloud["y"].to_numpy(),
        )
        xmin, xmax = float(ps["x"].min()), float(ps["x"].max())
        ymin, ymax = float(ps["y"].min()), float(ps["y"].max())
        if footprint_xy is not None and len(footprint_xy) > 3:
            xmin, xmax = float(footprint_xy["x"].min()), float(footprint_xy["x"].max())
            ymin, ymax = float(footprint_xy["y"].min()), float(footprint_xy["y"].max())
        interp_df = ps.copy()
        interp_df["disp_mm"] = ps_settlement
        interp_grid = sd.interpolated_grid(interp_df, xmin, xmax, ymin, ymax)
        sec_x = np.linspace(xmin, xmax, 220)
        sec_y = np.full_like(sec_x, float(ps["y"].median()))
        sec_settle = sd.idw_interpolate(ps["x"].to_numpy(), ps["y"].to_numpy(), ps_settlement, sec_x, sec_y)
        section_profile = (sec_x, sec_settle)
    else:
        ps = sd.insar_ps_points(b)
        ps_settlement = sd.settlement_field_mm(ps.assign(surface=0), nearest_insar_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
        ps_settlement += np.random.default_rng(123).normal(0.0, 0.6, size=len(ps_settlement))

    return {
        "idx": idx,
        "b": b,
        "cloud": cloud,
        "footprint_xy": footprint_xy,
        "sinkhole_xy": (sinkhole_x_m, sinkhole_y_m),
        "sigma_m": sigma_m,
        "fused_mm": fused_mm,
        "nearest_insar_mm": nearest_insar_mm,
        "settlement_mm": settlement_mm,
        "ps": ps,
        "ps_settlement": ps_settlement,
        "interp_grid": interp_grid,
        "section_profile": section_profile,
    }


def main() -> None:
    out_path = Path("outputs/settlement_dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = sd.retro_paths()
    retro_df, retro_summary = sd.retrospective_data()
    if retro_df is None or len(retro_df) == 0:
        raise RuntimeError(
            "No retrospective data found. Run eisenhower_insar_retrospective.py first."
        )

    state = _retrospective_visual_state(retro_df)
    idx = state["idx"]
    sinkhole_x_m, sinkhole_y_m = state["sinkhole_xy"]
    retro_figs = [
        sd.plan_view_fig(
            state["b"],
            sinkhole_x_m,
            sinkhole_y_m,
            state["sigma_m"],
            state["fused_mm"],
            sensors=None,
            footprint_xy=state["footprint_xy"],
            interp_grid=state["interp_grid"],
            sinkhole_marker=state["sinkhole_xy"],
        ),
        sd.section_view_fig(state["b"], sinkhole_x_m, state["sigma_m"], state["fused_mm"], section_profile=state["section_profile"]),
        sd.point_cloud_fig(
            state["cloud"],
            state["settlement_mm"],
            state["ps"],
            state["ps_settlement"],
            sensors=None,
            title="3D Point Cloud Projection: Eisenhower InSAR-only Retrospective",
        ),
        sd.insar_only_timeseries_fig(retro_df, idx),
        sd.risk_score_fig(retro_df, idx, retro_summary),
    ]
    divs = [fig_div(retro_figs[0], include_js=True)] + [fig_div(f, include_js=False) for f in retro_figs[1:]]

    summary_event = retro_summary.get("event_date", "2023-08-16")
    summary_alert = retro_summary.get("first_alert_date_in_claim_window", "n/a")
    summary_lead = retro_summary.get("lead_days_to_event", "n/a")
    summary_break = retro_summary.get("slope_break_date", "n/a")
    summary_break_p = retro_summary.get("slope_break_p_value", "n/a")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eisenhower Sinkhole Retrospective Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #102038; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    .header {{ background: #ffffff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; margin-top: 10px; }}
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
      <h2 style="margin:0 0 6px 0;">Eisenhower Sinkhole Retrospective (InSAR-only)</h2>
      <div>Static snapshot generated from upgraded retrospective outputs.</div>
      <div class="kpis">
        <div class="kpi"><div class="label">Selected Date</div><div class="value">{retro_df.loc[idx, "date"].date()}</div></div>
        <div class="kpi"><div class="label">Event Date</div><div class="value">{summary_event}</div></div>
        <div class="kpi"><div class="label">First Alert</div><div class="value">{summary_alert}</div></div>
        <div class="kpi"><div class="label">Lead Days</div><div class="value">{summary_lead}</div></div>
        <div class="kpi"><div class="label">Slope Break</div><div class="value">{summary_break}</div></div>
      </div>
      <div style="margin-top:8px;color:#35526f;font-size:13px;">Slope-break p-value: {summary_break_p}</div>
      <div style="margin-top:4px;color:#35526f;font-size:13px;">Data source directory: {paths["retro_dir"]}</div>
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
      <b>Interpretation:</b> This page is retrospective-first. Risk score blends velocity trend, cumulative settlement, short-vs-long acceleration energy, and Gaussian-bowl evidence from point-cloud displacement patterns.
    </div>
  </div>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
