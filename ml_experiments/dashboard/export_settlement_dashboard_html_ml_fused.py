#!/usr/bin/env python3
"""Export a standalone HTML version of the dashboard."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export standalone dashboard HTML.")
    p.add_argument("--mode", choices=["synthetic", "eisenhower"], default="eisenhower")
    p.add_argument("--out", default="outputs/settlement_dashboard.html")
    return p.parse_args()


def _render_synthetic(out_path: Path) -> None:
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
    divs = [fig_div(figs[0], include_js=True)] + [fig_div(f, include_js=False) for f in figs[1:]]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Settlement Structural Twin Dashboard (Synthetic)</title>
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h2 style="margin:0 0 6px 0;">Settlement-Focused Structural Digital Twin (Synthetic)</h2>
      <div class="kpis">
        <div class="kpi"><div class="label">Selected Date</div><div class="value">{df.loc[idx, "date"].date()}</div></div>
        <div class="kpi"><div class="label">Fused Settlement</div><div class="value">{fused_mm:.2f} mm</div></div>
        <div class="kpi"><div class="label">Nearest InSAR</div><div class="value">{nearest_insar_mm:.2f} mm</div></div>
        <div class="kpi"><div class="label">Modal Frequency</div><div class="value">{df.loc[idx, "modal_freq_hz"]:.3f} Hz</div></div>
      </div>
    </div>
    <div class="grid2"><div class="panel">{divs[0]}</div><div class="panel">{divs[1]}</div></div>
    <div class="panel">{divs[2]}</div>
    <div class="grid2"><div class="panel">{divs[3]}</div><div class="panel">{divs[4]}</div></div>
    <div class="note">Synthetic mode export.</div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _render_eisenhower(out_path: Path) -> None:
    retro_df, retro_summary = sd.retrospective_data()
    if retro_df is None or retro_df.empty:
        raise RuntimeError(f"Missing retrospective data at {sd.RETRO_CSV}")
    df = retro_df.copy().reset_index(drop=True)
    idx = len(df) - 1
    ml_ctx = sd.load_eisenhower_ml_context()

    real_geom = sd.load_eisenhower_geometry()
    if real_geom is not None:
        b = real_geom["building"]
        cloud = real_geom["cloud"]
        footprint_xy = real_geom["footprint"]
        real_ps_obs = sd.load_real_insar_points_local(real_geom["minx"], real_geom["miny"])
        sinkhole_xy_from_file = sd.load_sinkhole_location_local(real_geom["minx"], real_geom["miny"])
        if real_ps_obs is not None:
            ps = real_ps_obs[["x", "y"]].drop_duplicates().copy()
            ps["z"] = b.height_m
        else:
            ps = sd.insar_ps_points(b)
    else:
        b = sd.Building()
        cloud = sd.make_point_cloud(b)
        footprint_xy = None
        real_ps_obs = None
        sinkhole_xy_from_file = None
        ps = sd.insar_ps_points(b)

    if sinkhole_xy_from_file is not None:
        sinkhole_x_m, sinkhole_y_m = sinkhole_xy_from_file
    elif real_geom is not None:
        sinkhole_x_m, sinkhole_y_m = float(real_geom["sinkhole"]["x"]), float(real_geom["sinkhole"]["y"])
    else:
        sinkhole_x_m, sinkhole_y_m = 8.0, 15.0
    sigma_m = float(real_geom["sinkhole"]["sigma"]) if real_geom is not None else 6.0

    fused_mm = float(df.loc[idx, "smoothed_mm"])
    nearest_insar_mm = float(df.loc[idx, "displacement_mm"])
    ml_overlay = None
    fusion_state = None
    if ml_ctx is not None and ml_ctx.get("predictions") is not None:
        pred = ml_ctx["predictions"].copy()
        curr = pd.to_datetime(df.loc[idx, "date"], utc=True)
        if not pred.empty:
            before = pred[pred["date"] <= curr]
            row = before.iloc[-1] if not before.empty else pred.iloc[(pred["date"] - curr).abs().argmin()]
            event_dt = pd.to_datetime(retro_summary.get("event_date", str(sd.EVENT_DATE.date())), utc=True)
            lead_days = float((event_dt - curr).total_seconds() / 86400.0)
            ml_overlay = {
                "band": ml_ctx.get("best_band", "band_90_180"),
                "alert_state": bool(row.get("policy_pred", 0) == 1),
                "risk_prob": float(row.get("risk_prob")) if "risk_prob" in row else None,
                "lead_days": round(lead_days, 1),
            }
            risk_threshold = float(retro_summary.get("alert_threshold", 2.2))
            fusion_state = sd.fused_decision_state(
                risk_score=float(df.loc[idx, "risk_score"]),
                risk_threshold=risk_threshold,
                ml_alert=bool(ml_overlay["alert_state"]),
                ml_prob=ml_overlay.get("risk_prob"),
                rule="gate_and",
            )
            ml_overlay["fusion_rule"] = "gate_and"
            ml_overlay["fused_alert"] = bool(fusion_state.get("fused_alert", False))
    settlement_mm = sd.settlement_field_mm(cloud, fused_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
    interp_grid = None
    section_profile = None
    if real_ps_obs is not None:
        curr_date = pd.Timestamp(df.loc[idx, "date"])
        tmp = real_ps_obs.copy()
        tmp["d"] = (tmp["date"] - curr_date).abs()
        near = tmp.sort_values(["point_id", "d"]).groupby("point_id", as_index=False).first()
        ps = near[["x", "y"]].copy()
        ps["z"] = b.height_m
        ps_settlement = near["disp_mm"].to_numpy()
        settlement_mm = sd.idw_interpolate(
            ps["x"].to_numpy(), ps["y"].to_numpy(), ps_settlement, cloud["x"].to_numpy(), cloud["y"].to_numpy()
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
        ps_settlement = sd.settlement_field_mm(ps.assign(surface=0), nearest_insar_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
        ps_settlement += np.random.default_rng(123).normal(0.0, 0.6, size=len(ps_settlement))

    figs = [
        sd.plan_view_fig(
            b, sinkhole_x_m, sinkhole_y_m, sigma_m, fused_mm, None,
            footprint_xy=footprint_xy, interp_grid=interp_grid, sinkhole_marker=(sinkhole_x_m, sinkhole_y_m),
            ml_overlay=ml_overlay,
        ),
        sd.section_view_fig(b, sinkhole_x_m, sigma_m, fused_mm, section_profile=section_profile),
        sd.point_cloud_fig(
            cloud,
            settlement_mm,
            ps,
            ps_settlement,
            None,
            title=(
                "3D Point Cloud Projection: Eisenhower InSAR-only Retrospective"
                if ml_overlay is None
                else "3D Point Cloud Projection: Eisenhower InSAR + ML Context "
                f"[{ml_overlay.get('band')} | "
                f"{'ALERT' if ml_overlay.get('alert_state') else 'NO ALERT'} | "
                f"lead {ml_overlay.get('lead_days')} d]"
            ),
        ),
        sd.insar_only_timeseries_fig(df, idx),
        sd.risk_score_fig(df, idx, retro_summary),
    ]
    divs = [fig_div(figs[0], include_js=True)] + [fig_div(f, include_js=False) for f in figs[1:]]

    first_alert = retro_summary.get("first_alert_date_in_claim_window", "n/a")
    lead_days = retro_summary.get("lead_days_to_event", "n/a")
    event_date = retro_summary.get("event_date", "2023-08-16")
    threshold = retro_summary.get("alert_threshold", "n/a")
    ml_band = "n/a" if ml_overlay is None else ml_overlay.get("band", "n/a")
    ml_alert = "n/a" if ml_overlay is None else ("ON" if ml_overlay.get("alert_state", False) else "OFF")
    ml_lead = "n/a" if ml_overlay is None else ml_overlay.get("lead_days", "n/a")
    fused_alert = "n/a" if fusion_state is None else ("ON" if fusion_state.get("fused_alert", False) else "OFF")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eisenhower Retrospective Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #102038; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    .header {{ background: #ffffff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(9, minmax(105px, 1fr)); gap: 10px; margin-top: 10px; }}
    .kpi {{ background: #eff4fb; border: 1px solid #d8e0eb; border-radius: 10px; padding: 10px; }}
    .kpi .label {{ font-size: 12px; color: #37506d; }}
    .kpi .value {{ font-weight: 700; font-size: 18px; margin-top: 3px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ background: #fff; border: 1px solid #d8e0eb; border-radius: 12px; padding: 8px; margin-bottom: 12px; }}
    .note {{ background: #fffdf3; border: 1px solid #f0dfaa; border-radius: 12px; padding: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h2 style="margin:0 0 6px 0;">Eisenhower Retrospective (InSAR-only)</h2>
      <div class="kpis">
        <div class="kpi"><div class="label">Selected Date</div><div class="value">{df.loc[idx, "date"].date()}</div></div>
        <div class="kpi"><div class="label">Smoothed InSAR</div><div class="value">{fused_mm:.2f} mm</div></div>
        <div class="kpi"><div class="label">Risk Score</div><div class="value">{float(df.loc[idx, "risk_score"]):.2f}</div></div>
        <div class="kpi"><div class="label">First Alert</div><div class="value">{first_alert}</div></div>
        <div class="kpi"><div class="label">Lead Days</div><div class="value">{lead_days}</div></div>
        <div class="kpi"><div class="label">ML Band</div><div class="value">{ml_band}</div></div>
        <div class="kpi"><div class="label">ML Alert</div><div class="value">{ml_alert}</div></div>
        <div class="kpi"><div class="label">ML Lead Days</div><div class="value">{ml_lead}</div></div>
        <div class="kpi"><div class="label">Fused Alert</div><div class="value">{fused_alert}</div></div>
      </div>
    </div>
    <div class="grid2"><div class="panel">{divs[0]}</div><div class="panel">{divs[1]}</div></div>
    <div class="panel">{divs[2]}</div>
    <div class="grid2"><div class="panel">{divs[3]}</div><div class="panel">{divs[4]}</div></div>
    <div class="note">
      Event date: {event_date}. Alert threshold: {threshold}. Source: {retro_summary.get("source", "unknown")}.
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "synthetic":
        _render_synthetic(out_path)
    else:
        _render_eisenhower(out_path)
    print(out_path.resolve())


if __name__ == "__main__":
    main()
