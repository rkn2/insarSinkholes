#!/usr/bin/env python3
"""Streamlit dashboard for synthetic sinkhole/settlement structural twin monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyproj import Transformer
from scipy.spatial import cKDTree
from plotly.subplots import make_subplots
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon

from synthetic_structural_twin_demo import (
    Config,
    kalman_fuse,
    make_true_state,
    simulate_accel,
    simulate_insar,
)

EVENT_DATE = pd.Timestamp("2023-08-16")
RETRO_CSV = Path(
    "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/eisenhower_retrospective/insar_retrospective_timeseries.csv"
)
RETRO_SUMMARY_JSON = Path(
    "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/eisenhower_retrospective/retrospective_summary.json"
)
RETRO_POINT_OBS_CSV = Path(
    "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/eisenhower_retrospective/insar_point_observations.csv"
)
SINKHOLE_LOCATION_CSV = Path(
    "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/eisenhower_retrospective/sinkhole_location.csv"
)
EISENHOWER_EVENT_ID = "2023-08-16_eisenhower_parking_deck_penn_state"
LOEO_REAL_CONTROLS_DIR = Path(
    "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/ml/loeo_real_controls_v1"
)
LOEO_RUN_DIRS = {
    "LOEO Real Controls v1": Path(
        "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/ml/loeo_real_controls_v1"
    ),
    "LOEO Hardened v1 (synthetic controls)": Path(
        "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/ml/loeo_hardened_v1"
    ),
    "LOEO Baseline v1": Path(
        "/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/ml/loeo_v1"
    ),
}


@dataclass
class Building:
    length_m: float = 60.0
    width_m: float = 30.0
    height_m: float = 18.0


def synthetic_timeseries(seed: int = 7) -> pd.DataFrame:
    cfg = Config(seed=seed)
    rng = np.random.default_rng(seed)
    state = make_true_state(cfg, rng)
    insar = simulate_insar(cfg, state, rng)
    accel_proxy, modal_freq = simulate_accel(cfg, state, rng)
    fused = kalman_fuse(accel_proxy, insar)
    df = state.copy()
    df["insar_mm"] = insar
    df["accel_proxy_mm"] = accel_proxy
    df["modal_freq_hz"] = modal_freq
    df["fused_mm"] = fused
    return df


def make_point_cloud(b: Building, nxy: int = 36, nz: int = 18) -> pd.DataFrame:
    x = np.linspace(0.0, b.length_m, nxy)
    y = np.linspace(0.0, b.width_m, nxy)
    z = np.linspace(0.0, b.height_m, nz)

    X, Y = np.meshgrid(x, y)
    roof = np.c_[X.ravel(), Y.ravel(), np.full(X.size, b.height_m), np.full(X.size, 1)]
    base = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size), np.full(X.size, 0)]

    Xw, Zw = np.meshgrid(x, z)
    wall1 = np.c_[Xw.ravel(), np.zeros(Xw.size), Zw.ravel(), np.full(Xw.size, 2)]
    wall2 = np.c_[Xw.ravel(), np.full(Xw.size, b.width_m), Zw.ravel(), np.full(Xw.size, 2)]

    Yw, Zw2 = np.meshgrid(y, z)
    wall3 = np.c_[np.zeros(Yw.size), Yw.ravel(), Zw2.ravel(), np.full(Yw.size, 2)]
    wall4 = np.c_[np.full(Yw.size, b.length_m), Yw.ravel(), Zw2.ravel(), np.full(Yw.size, 2)]

    pts = np.vstack([roof, base, wall1, wall2, wall3, wall4])
    return pd.DataFrame(pts, columns=["x", "y", "z", "surface"])


def _largest_polygon(geom: Polygon | MultiPolygon) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    return max(geom.geoms, key=lambda g: g.area)


@st.cache_data(show_spinner=False)
def load_eisenhower_geometry() -> dict | None:
    lat, lon = 40.8023011, -77.8609059
    gdf = ox.features_from_point((lat, lon), tags={"building": True}, dist=300)
    if gdf.empty:
        return None
    names = gdf.get("name")
    if names is None:
        return None
    mask = names.fillna("").str.contains("Eisenhower Parking Deck", case=False)
    cand = gdf[mask].copy()
    if cand.empty:
        return None
    cand = cand.to_crs(3857)
    geom = _largest_polygon(cand.geometry.iloc[0])
    minx, miny, maxx, maxy = geom.bounds
    geom_local = affinity.translate(geom, xoff=-minx, yoff=-miny)

    building = Building(length_m=float(maxx - minx), width_m=float(maxy - miny), height_m=18.0)
    cloud = make_point_cloud_from_polygon(geom_local, building.height_m)
    xy = np.array(geom_local.exterior.coords)
    footprint = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})

    # Placeholder sinkhole center near "front" edge, inferred from road-side edge not official failure survey.
    target = Point(building.length_m, 0.5 * building.width_m)
    edge_pt = geom_local.exterior.interpolate(geom_local.exterior.project(target))
    sinkhole = {"x": float(edge_pt.x), "y": float(edge_pt.y), "sigma": 6.0}
    return {"building": building, "cloud": cloud, "footprint": footprint, "sinkhole": sinkhole, "minx": float(minx), "miny": float(miny)}


@st.cache_data(show_spinner=False)
def load_real_insar_points_local(minx: float, miny: float) -> pd.DataFrame | None:
    if not RETRO_POINT_OBS_CSV.exists():
        return None
    df = pd.read_csv(RETRO_POINT_OBS_CSV)
    required = {"lon", "lat", "date", "disp_m"}
    if not required.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "lon", "lat", "disp_m"]).copy()
    if df.empty:
        return None
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = t.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"] = x - minx
    df["y"] = y - miny
    df["disp_mm"] = pd.to_numeric(df["disp_m"], errors="coerce") * 1000.0
    return df


@st.cache_data(show_spinner=False)
def load_sinkhole_location_local(minx: float, miny: float) -> tuple[float, float] | None:
    if not SINKHOLE_LOCATION_CSV.exists():
        return None
    df = pd.read_csv(SINKHOLE_LOCATION_CSV)
    cols = {c.strip().lower(): c for c in df.columns}
    if "lat" not in cols or "lon" not in cols:
        return None
    lat = float(df.iloc[0][cols["lat"]])
    lon = float(df.iloc[0][cols["lon"]])
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = t.transform(lon, lat)
    return float(x - minx), float(y - miny)


def make_point_cloud_from_polygon(poly: Polygon, height_m: float, nxy: int = 70, nz: int = 18) -> pd.DataFrame:
    minx, miny, maxx, maxy = poly.bounds
    xg = np.linspace(minx, maxx, nxy)
    yg = np.linspace(miny, maxy, nxy)
    inside = []
    for x in xg:
        for y in yg:
            if poly.contains(Point(float(x), float(y))):
                inside.append((x, y))
    inside_arr = np.array(inside, dtype=float)
    if len(inside_arr) == 0:
        return make_point_cloud(Building(), nxy=36, nz=nz)

    roof = np.c_[inside_arr[:, 0], inside_arr[:, 1], np.full(len(inside_arr), height_m), np.full(len(inside_arr), 1)]
    base = np.c_[inside_arr[:, 0], inside_arr[:, 1], np.zeros(len(inside_arr)), np.full(len(inside_arr), 0)]

    edge = np.array(poly.exterior.coords)
    wall_pts = []
    zvals = np.linspace(0.0, height_m, nz)
    for i in range(len(edge) - 1):
        p1, p2 = edge[i], edge[i + 1]
        seg_len = np.hypot(*(p2 - p1))
        nseg = max(2, int(seg_len / 2.0))
        s = np.linspace(0.0, 1.0, nseg)
        xs = p1[0] + (p2[0] - p1[0]) * s
        ys = p1[1] + (p2[1] - p1[1]) * s
        for x, y in zip(xs, ys):
            for z in zvals:
                wall_pts.append((x, y, z, 2))
    walls = np.array(wall_pts, dtype=float)
    pts = np.vstack([roof, base, walls])
    return pd.DataFrame(pts, columns=["x", "y", "z", "surface"])


def settlement_field_mm(
    pts: pd.DataFrame,
    base_mm: float,
    sinkhole_x_m: float,
    sinkhole_y_m: float,
    sigma_m: float,
) -> np.ndarray:
    dx = pts["x"].to_numpy() - sinkhole_x_m
    dy = pts["y"].to_numpy() - sinkhole_y_m
    r2 = dx**2 + dy**2
    bowl = np.exp(-r2 / (2.0 * sigma_m**2))
    depth_factor = 1.0 - 0.65 * (pts["z"].to_numpy() / pts["z"].max())
    return base_mm * (0.45 + 0.85 * bowl) * depth_factor


def idw_interpolate(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    x_q: np.ndarray,
    y_q: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    if len(x_obs) == 0:
        return np.zeros_like(x_q, dtype=float)
    tree = cKDTree(np.c_[x_obs, y_obs])
    k_eff = max(1, min(k, len(x_obs)))
    d, idx = tree.query(np.c_[x_q, y_q], k=k_eff)
    if k_eff == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(d, 1e-6) ** power
    z_nei = z_obs[idx]
    return (w * z_nei).sum(axis=1) / w.sum(axis=1)


def interpolated_grid(
    points_xy_mm: pd.DataFrame,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int = 90,
    ny: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    Z = idw_interpolate(
        points_xy_mm["x"].to_numpy(),
        points_xy_mm["y"].to_numpy(),
        points_xy_mm["disp_mm"].to_numpy(),
        X.ravel(),
        Y.ravel(),
    ).reshape(Y.shape)
    return x, y, Z


def insar_ps_points(b: Building, n: int = 90, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-20, b.length_m + 20, size=n)
    ys = rng.uniform(-20, b.width_m + 20, size=n)
    zs = np.where(
        (xs >= 0) & (xs <= b.length_m) & (ys >= 0) & (ys <= b.width_m),
        b.height_m,
        0.0,
    )
    return pd.DataFrame({"x": xs, "y": ys, "z": zs})


def accel_sensors(b: Building) -> pd.DataFrame:
    pts = np.array(
        [
            [0.0, 0.0, 0.0, "A1 Corner SW"],
            [b.length_m, 0.0, 0.0, "A2 Corner SE"],
            [b.length_m, b.width_m, 0.0, "A3 Corner NE"],
            [0.0, b.width_m, 0.0, "A4 Corner NW"],
            [0.5 * b.length_m, 0.5 * b.width_m, 0.0, "A5 Core"],
        ],
        dtype=object,
    )
    return pd.DataFrame(pts, columns=["x", "y", "z", "sensor"])


def nearest_insar_value(series: pd.Series, idx: int) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    nearest_i = np.abs(valid.index.to_numpy() - idx).argmin()
    return float(valid.iloc[nearest_i])


def plan_view_fig(
    b: Building,
    sinkhole_x_m: float,
    sinkhole_y_m: float,
    sigma_m: float,
    base_mm: float,
    sensors: pd.DataFrame | None = None,
    footprint_xy: pd.DataFrame | None = None,
    interp_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    sinkhole_marker: tuple[float, float] | None = None,
    ml_overlay: dict | None = None,
) -> go.Figure:
    if footprint_xy is not None and len(footprint_xy) > 3:
        xmin, xmax = float(footprint_xy["x"].min()), float(footprint_xy["x"].max())
        ymin, ymax = float(footprint_xy["y"].min()), float(footprint_xy["y"].max())
    else:
        xmin, xmax, ymin, ymax = 0.0, b.length_m, 0.0, b.width_m
    if interp_grid is None:
        x = np.linspace(xmin, xmax, 80)
        y = np.linspace(ymin, ymax, 50)
        X, Y = np.meshgrid(x, y)
        r2 = (X - sinkhole_x_m) ** 2 + (Y - sinkhole_y_m) ** 2
        Z = base_mm * (0.45 + 0.85 * np.exp(-r2 / (2.0 * sigma_m**2)))
    else:
        x, y, Z = interp_grid

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=Z,
            colorscale="YlOrRd",
            colorbar={"title": "mm"},
            name="Settlement map",
        )
    )
    if footprint_xy is not None and len(footprint_xy) > 3:
        fig.add_trace(
            go.Scatter(
                x=footprint_xy["x"],
                y=footprint_xy["y"],
                mode="lines",
                line={"color": "black", "width": 2},
                name="Deck footprint (OSM)",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0, b.length_m, b.length_m, 0, 0],
                y=[0, 0, b.width_m, b.width_m, 0],
                mode="lines",
                line={"color": "black", "width": 2},
                name="Building plan",
            )
        )
    if sensors is not None and len(sensors) > 0:
        fig.add_trace(
            go.Scatter(
                x=sensors["x"],
                y=sensors["y"],
                mode="markers+text",
                text=sensors["sensor"],
                textposition="top center",
                marker={"size": 10, "color": "cyan", "line": {"color": "black", "width": 1}},
                name="Accelerometers",
            )
        )
    if sinkhole_marker is not None:
        marker_color = "magenta"
        marker_label = "Reported sinkhole location"
        if ml_overlay is not None:
            if bool(ml_overlay.get("alert_state", False)):
                marker_color = "red"
            else:
                marker_color = "orange"
            marker_label = (
                f"ML {ml_overlay.get('band', 'band')} "
                f"{'ALERT' if ml_overlay.get('alert_state', False) else 'NO ALERT'}"
            )
        fig.add_trace(
            go.Scatter(
                x=[sinkhole_marker[0]],
                y=[sinkhole_marker[1]],
                mode="markers+text",
                text=[marker_label],
                textposition="top center",
                marker={"size": 14, "color": marker_color, "symbol": "x", "line": {"color": "white", "width": 1}},
                name="Sinkhole (reported)",
            )
        )
    fig.update_layout(
        title="Plan View: Foundation Settlement Hotspots",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    if ml_overlay is not None:
        risk_prob = ml_overlay.get("risk_prob")
        rp = "n/a" if risk_prob is None or pd.isna(risk_prob) else f"{float(risk_prob):.3f}"
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            text=(
                f"ML band: {ml_overlay.get('band', 'n/a')} | "
                f"state: {'ALERT' if ml_overlay.get('alert_state', False) else 'NO ALERT'} | "
                f"risk_prob: {rp} | lead: {ml_overlay.get('lead_days', 'n/a')} | "
                f"fused: {'ALERT' if ml_overlay.get('fused_alert', False) else 'NO ALERT'}"
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font={"size": 11},
        )
    return fig


def section_view_fig(
    b: Building,
    sinkhole_x_m: float,
    sigma_m: float,
    base_mm: float,
    section_profile: tuple[np.ndarray, np.ndarray] | None = None,
) -> go.Figure:
    if section_profile is None:
        x = np.linspace(0.0, b.length_m, 220)
        r2 = (x - sinkhole_x_m) ** 2
        settlement = base_mm * (0.45 + 0.85 * np.exp(-r2 / (2.0 * sigma_m**2)))
    else:
        x, settlement = section_profile
    deformed_base = -settlement / 1000.0
    roof = np.full_like(x, b.height_m)
    deformed_roof = roof - 0.35 * settlement / 1000.0

    # Split-axis section view to remove empty middle and expose mm-level variation.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.48, 0.52],
        subplot_titles=("Roof band", "Foundation band"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=roof, mode="lines", line={"color": "gray", "dash": "dot"}, name="Original roof"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=deformed_roof, mode="lines", line={"color": "orange", "width": 2}, name="Deformed roof"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=np.zeros_like(x), mode="lines", line={"color": "gray", "dash": "dot"}, name="Original base"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=deformed_base, mode="lines", line={"color": "red", "width": 3}, name="Deformed base"),
        row=2,
        col=1,
    )
    roof_pad = max(0.03, float(np.ptp(deformed_roof)) * 4.0)
    base_pad = max(0.03, float(np.ptp(deformed_base)) * 4.0)
    fig.update_yaxes(title_text="Elevation (m)", range=[float(deformed_roof.min() - roof_pad), float(roof.max() + roof_pad)], row=1, col=1)
    fig.update_yaxes(title_text="Elevation (m)", range=[float(deformed_base.min() - base_pad), float(base_pad)], row=2, col=1)
    fig.update_xaxes(title_text="X (m)", row=2, col=1)
    fig.update_layout(
        title="Section View: Vertical Deformation Profile (Split Axis)",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def point_cloud_fig(
    cloud: pd.DataFrame,
    settlement_mm: np.ndarray,
    insar_points: pd.DataFrame,
    insar_mm: np.ndarray,
    sensors: pd.DataFrame | None = None,
    title: str = "3D Point Cloud Projection: Settlement from InSAR + Accelerometer Fusion",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=cloud["x"],
            y=cloud["y"],
            z=cloud["z"] - settlement_mm / 1000.0,
            mode="markers",
            marker={
                "size": 2.5,
                "color": settlement_mm,
                "colorscale": "Turbo",
                "colorbar": {"title": "Projected settlement (mm)"},
                "opacity": 0.75,
            },
            name="Building point cloud",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=insar_points["x"],
            y=insar_points["y"],
            z=insar_points["z"] + 1.2,
            mode="markers",
            marker={"size": 4, "color": insar_mm, "colorscale": "Viridis", "opacity": 0.9},
            name="InSAR PS",
        )
    )
    if sensors is not None and len(sensors) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=sensors["x"],
                y=sensors["y"],
                z=sensors["z"] + 0.5,
                mode="markers+text",
                text=sensors["sensor"],
                textposition="top center",
                marker={"size": 6, "color": "cyan", "line": {"color": "black", "width": 1}},
                name="Accelerometers",
            )
        )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def time_series_fig(df: pd.DataFrame, idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["insar_mm"], mode="markers+lines", name="InSAR LOS displacement (mm)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["accel_proxy_mm"], mode="lines", name="Accel-derived displacement proxy (mm)", opacity=0.5))
    fig.add_trace(go.Scatter(x=df["date"], y=df["fused_mm"], mode="lines", name="Fused settlement estimate (mm)", line={"width": 3}))
    fig.add_vline(x=df.loc[idx, "date"], line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Settlement Time Series",
        xaxis_title="Date",
        yaxis_title="mm",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def modal_fig(df: pd.DataFrame, idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["modal_freq_hz"], mode="lines", name="Estimated fundamental frequency (Hz)"))
    fig.add_vline(x=df.loc[idx, "date"], line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Accelerometer Modal Indicator",
        xaxis_title="Date",
        yaxis_title="Hz",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def retrospective_data() -> tuple[pd.DataFrame | None, dict]:
    if not RETRO_CSV.exists():
        return None, {}
    df = pd.read_csv(RETRO_CSV)
    if "date" not in df.columns:
        return None, {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    summary = {}
    if RETRO_SUMMARY_JSON.exists():
        import json

        summary = json.loads(RETRO_SUMMARY_JSON.read_text(encoding="utf-8"))
    return df, summary


@st.cache_data(show_spinner=False)
def load_loeo_bundle(base_dir: Path) -> dict | None:
    if not base_dir.exists():
        return None
    agg_all = base_dir / "loeo_aggregate_summary_all_bands.csv"
    evt_all = base_dir / "loeo_event_summary_all_bands.csv"
    bench_json = base_dir / "frozen_benchmark_report.json"
    if agg_all.exists() and evt_all.exists():
        agg_df = pd.read_csv(agg_all)
        evt_df = pd.read_csv(evt_all)
        bench = {}
        if bench_json.exists():
            bench = json.loads(bench_json.read_text(encoding="utf-8"))
        return {"aggregate": agg_df, "events": evt_df, "benchmark": bench, "mode": "multiband"}

    # Backward compatibility for older single-band LOEO outputs.
    agg_single = base_dir / "loeo_aggregate_summary.json"
    evt_single = base_dir / "loeo_event_summary.csv"
    if agg_single.exists() and evt_single.exists():
        agg = json.loads(agg_single.read_text(encoding="utf-8"))
        agg_df = pd.DataFrame([agg])
        agg_df["band"] = "single_band"
        evt_df = pd.read_csv(evt_single)
        evt_df["band"] = "single_band"
        return {"aggregate": agg_df, "events": evt_df, "benchmark": {}, "mode": "single"}
    return None


@st.cache_data(show_spinner=False)
def load_eisenhower_ml_context(base_dir: Path = LOEO_REAL_CONTROLS_DIR) -> dict | None:
    if not base_dir.exists():
        return None
    bench_path = base_dir / "frozen_benchmark_report.json"
    if not bench_path.exists():
        return None

    try:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    bands = bench.get("bands", [])
    if not bands:
        return None
    best = max(bands, key=lambda r: float(r.get("benchmark_policy_f1", 0.0)))
    best_band = str(best.get("band", "band_90_180"))

    fold_name = f"05_{EISENHOWER_EVENT_ID}"
    pred_path = base_dir / best_band / "folds" / fold_name / "classifier" / "test_policy_predictions.csv"
    if not pred_path.exists():
        return {"best_band": best_band, "benchmark": best, "predictions": None}

    df = pd.read_csv(pred_path)
    if "event_id" not in df.columns or "date" not in df.columns:
        return {"best_band": best_band, "benchmark": best, "predictions": None}
    df = df[df["event_id"] == EISENHOWER_EVENT_ID].copy()
    if df.empty:
        return {"best_band": best_band, "benchmark": best, "predictions": None}
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)
    return {"best_band": best_band, "benchmark": best, "predictions": df}


def insar_only_timeseries_fig(df: pd.DataFrame, idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["displacement_mm"], mode="markers+lines", name="InSAR displacement (obs)", opacity=0.6))
    fig.add_trace(go.Scatter(x=df["date"], y=df["smoothed_mm"], mode="lines", name="Smoothed displacement", line={"width": 3}))
    if "changepoint" in df.columns:
        cp = df[df["changepoint"].astype(bool)]
        if len(cp):
            fig.add_trace(go.Scatter(x=cp["date"], y=cp["smoothed_mm"], mode="markers", name="Detected changepoint", marker={"color": "red", "size": 8}))
    fig.add_vline(x=EVENT_DATE, line_width=2, line_dash="dash", line_color="black")
    fig.add_vline(x=df.loc[idx, "date"], line_width=2, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Eisenhower InSAR Displacement Timeline",
        xaxis_title="Date",
        yaxis_title="LOS displacement (mm)",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def risk_score_fig(df: pd.DataFrame, idx: int, summary: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["risk_score"], mode="lines", name="Risk score", line={"width": 3, "color": "orange"}))
    if "velocity_mm_per_day" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["velocity_mm_per_day"],
                mode="lines",
                name="Raw velocity (mm/day)",
                line={"width": 2, "color": "steelblue"},
                yaxis="y2",
            )
        )
    threshold = float(summary.get("alert_threshold", 2.2))
    fig.add_hline(y=threshold, line_width=2, line_dash="dash", line_color="red")
    event_dt = pd.Timestamp(summary.get("event_date", str(EVENT_DATE.date())))
    fig.add_vline(x=event_dt, line_width=2, line_dash="dash", line_color="black")
    fig.add_vline(x=df.loc[idx, "date"], line_width=2, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Precursor Risk Score + Raw Velocity (InSAR-only)",
        xaxis_title="Date",
        yaxis_title="score",
        yaxis2={"title": "mm/day", "overlaying": "y", "side": "right", "showgrid": False},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


def fused_decision_state(
    risk_score: float,
    risk_threshold: float,
    ml_alert: bool | None,
    ml_prob: float | None,
    rule: str,
    alpha: float = 0.6,
    blend_threshold: float = 0.95,
) -> dict:
    risk_on = bool(risk_score >= risk_threshold)
    if ml_alert is None and ml_prob is None:
        return {"fused_alert": risk_on, "risk_on": risk_on, "ml_on": None, "fused_score": None}
    ml_on = bool(ml_alert) if ml_alert is not None else bool((ml_prob or 0.0) >= 0.5)

    if rule == "gate_and":
        return {"fused_alert": bool(risk_on and ml_on), "risk_on": risk_on, "ml_on": ml_on, "fused_score": None}

    norm_risk = float(risk_score / max(1e-9, risk_threshold))
    norm_risk = max(0.0, min(norm_risk, 2.0))
    prob = float(ml_prob if ml_prob is not None else (1.0 if ml_on else 0.0))
    fused_score = float(alpha) * norm_risk + (1.0 - float(alpha)) * prob
    fused_alert = fused_score >= float(blend_threshold)
    return {"fused_alert": bool(fused_alert), "risk_on": risk_on, "ml_on": ml_on, "fused_score": fused_score}


def main() -> None:
    st.set_page_config(page_title="Settlement Digital Twin Dashboard", layout="wide")
    st.title("Settlement-Focused Structural Digital Twin Dashboard")
    st.caption("Includes synthetic fusion mode and Eisenhower retrospective InSAR-only mode.")

    b = Building()
    cloud = make_point_cloud(b)
    footprint_xy = None
    ps = insar_ps_points(b)
    real_ps_obs = None
    sinkhole_xy_from_file = None
    retro_df, retro_summary = retrospective_data()
    ml_ctx = load_eisenhower_ml_context()
    real_geom = load_eisenhower_geometry()

    st.sidebar.header("Scenario Controls")
    mode_options = ["Synthetic Fusion Demo"]
    if retro_df is not None:
        mode_options.append("Eisenhower Retrospective (InSAR-only)")
    mode = st.sidebar.radio("Dataset mode", options=mode_options, index=len(mode_options) - 1)

    if mode == "Eisenhower Retrospective (InSAR-only)":
        df = retro_df.copy()
        sensors = None
        if real_geom is not None:
            b = real_geom["building"]
            cloud = real_geom["cloud"]
            footprint_xy = real_geom["footprint"]
            real_ps_obs = load_real_insar_points_local(real_geom["minx"], real_geom["miny"])
            sinkhole_xy_from_file = load_sinkhole_location_local(real_geom["minx"], real_geom["miny"])
            if real_ps_obs is not None:
                ps = real_ps_obs[["x", "y"]].drop_duplicates().copy()
                ps["z"] = b.height_m
            else:
                ps = insar_ps_points(b)
        min_d = pd.Timestamp(df["date"].min()).date()
        max_d = pd.Timestamp(df["date"].max()).date()
        start_d = st.sidebar.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
        end_d = st.sidebar.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)
        if start_d > end_d:
            st.sidebar.error("Start date must be before end date.")
            st.stop()
        df = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].reset_index(drop=True)
        if df.empty:
            st.warning("No observations in selected date range.")
            st.stop()
    else:
        df = synthetic_timeseries(seed=7)
        sensors = accel_sensors(b)

    idx = st.sidebar.slider("Date index", min_value=0, max_value=len(df) - 1, value=len(df) - 1)
    ml_overlay = None
    fusion_state = None
    if mode == "Eisenhower Retrospective (InSAR-only)":
        st.sidebar.caption("Sinkhole geometry is fixed in retrospective mode.")
        default_x = (
            float(sinkhole_xy_from_file[0])
            if sinkhole_xy_from_file is not None
            else (
                float(real_geom["sinkhole"]["x"])
                if real_geom is not None
                else 8.0
            )
        )
        default_y = (
            float(sinkhole_xy_from_file[1])
            if sinkhole_xy_from_file is not None
            else 8.0
        )
        if sinkhole_xy_from_file is None and real_geom is not None:
            default_y = float(real_geom["sinkhole"]["y"])
        elif sinkhole_xy_from_file is None:
            default_y = 15.0
        default_sigma = (
            float(real_geom["sinkhole"]["sigma"])
            if real_geom is not None
            else 6.0
        )
        sinkhole_x_m = st.sidebar.slider(
            "Sinkhole center X (m)",
            min_value=0.0,
            max_value=b.length_m,
            value=float(np.clip(default_x, 0.0, b.length_m)),
            disabled=True,
        )
        sinkhole_y_m = st.sidebar.slider(
            "Sinkhole center Y (m)",
            min_value=0.0,
            max_value=b.width_m,
            value=float(np.clip(default_y, 0.0, b.width_m)),
            disabled=True,
        )
        sigma_m = st.sidebar.slider(
            "Sinkhole influence radius (sigma, m)",
            min_value=2.0,
            max_value=20.0,
            value=float(np.clip(default_sigma, 2.0, 20.0)),
            disabled=True,
        )
        scale = st.sidebar.slider("Settlement amplification", min_value=0.4, max_value=2.0, value=1.0, disabled=True)
    else:
        sinkhole_x_m = st.sidebar.slider("Sinkhole center X (m)", min_value=0.0, max_value=b.length_m, value=24.0)
        sinkhole_y_m = st.sidebar.slider("Sinkhole center Y (m)", min_value=0.0, max_value=b.width_m, value=12.0)
        sigma_m = st.sidebar.slider("Sinkhole influence radius (sigma, m)", min_value=2.0, max_value=20.0, value=8.0)
        scale = st.sidebar.slider("Settlement amplification", min_value=0.4, max_value=2.0, value=1.0)

    if mode == "Eisenhower Retrospective (InSAR-only)":
        fused_mm = float(df.loc[idx, "smoothed_mm"]) * scale
        nearest_insar_mm = float(df.loc[idx, "displacement_mm"]) * scale
        if ml_ctx is not None and ml_ctx.get("predictions") is not None:
            pred = ml_ctx["predictions"].copy()
            curr = pd.Timestamp(df.loc[idx, "date"], tz="UTC")
            if not pred.empty:
                before = pred[pred["date"] <= curr]
                row = before.iloc[-1] if not before.empty else pred.iloc[(pred["date"] - curr).abs().argmin()]
                event_dt = pd.Timestamp(retro_summary.get("event_date", str(EVENT_DATE.date())), tz="UTC")
                lead_days = float((event_dt - curr).total_seconds() / 86400.0)
                ml_overlay = {
                    "band": ml_ctx.get("best_band", "band_90_180"),
                    "alert_state": bool(row.get("policy_pred", 0) == 1),
                    "risk_prob": float(row.get("risk_prob")) if "risk_prob" in row else None,
                    "lead_days": round(lead_days, 1),
                }
        fusion_rule = st.sidebar.selectbox(
            "Fused decision rule",
            options=["gate_and", "weighted_blend"],
            index=0,
            help="gate_and: risk>=threshold AND ML alert. weighted_blend: alpha*(risk/threshold)+(1-alpha)*ML_prob >= blend threshold.",
        )
        alpha = st.sidebar.slider("Blend alpha (risk weight)", 0.0, 1.0, 0.6, 0.05) if fusion_rule == "weighted_blend" else 0.6
        blend_threshold = (
            st.sidebar.slider("Blend alert threshold", 0.5, 1.5, 0.95, 0.05) if fusion_rule == "weighted_blend" else 0.95
        )
        risk_threshold = float(retro_summary.get("alert_threshold", 2.2))
        fusion_state = fused_decision_state(
            risk_score=float(df.loc[idx, "risk_score"]),
            risk_threshold=risk_threshold,
            ml_alert=None if ml_overlay is None else bool(ml_overlay.get("alert_state", False)),
            ml_prob=None if ml_overlay is None else ml_overlay.get("risk_prob"),
            rule=fusion_rule,
            alpha=alpha,
            blend_threshold=blend_threshold,
        )
        if ml_overlay is not None:
            ml_overlay["fusion_rule"] = fusion_rule
            ml_overlay["fused_alert"] = bool(fusion_state.get("fused_alert", False))
    else:
        fused_mm = float(df.loc[idx, "fused_mm"]) * scale
        nearest_insar_mm = nearest_insar_value(df["insar_mm"], idx) * scale

    settlement_mm = settlement_field_mm(cloud, fused_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
    interp_grid = None
    section_profile = None
    if mode == "Eisenhower Retrospective (InSAR-only)" and real_ps_obs is not None:
        curr_date = pd.Timestamp(df.loc[idx, "date"])
        tmp = real_ps_obs.copy()
        tmp["d"] = (tmp["date"] - curr_date).abs()
        near = (
            tmp.sort_values(["point_id", "d"])
            .groupby("point_id", as_index=False)
            .first()
        )
        ps = near[["x", "y"]].copy()
        ps["z"] = b.height_m
        ps_settlement = near["disp_mm"].to_numpy() * scale
        # Data-driven interpolation from real point displacements.
        settlement_mm = idw_interpolate(
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
        interp_grid = interpolated_grid(interp_df, xmin, xmax, ymin, ymax)
        sec_x = np.linspace(xmin, xmax, 220)
        sec_y = np.full_like(sec_x, float(ps["y"].median()))
        sec_settle = idw_interpolate(
            ps["x"].to_numpy(),
            ps["y"].to_numpy(),
            ps_settlement,
            sec_x,
            sec_y,
        )
        section_profile = (sec_x, sec_settle)
    else:
        ps_settlement = settlement_field_mm(ps.assign(surface=0), nearest_insar_mm, sinkhole_x_m, sinkhole_y_m, sigma_m)
        ps_settlement += np.random.default_rng(123).normal(0.0, 0.6, size=len(ps_settlement))

    if mode == "Eisenhower Retrospective (InSAR-only)":
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Selected date", f"{df.loc[idx, 'date'].date()}")
    if mode == "Eisenhower Retrospective (InSAR-only)":
        kpi2.metric("Smoothed InSAR", f"{fused_mm:.2f} mm")
        kpi3.metric("Observed InSAR", f"{nearest_insar_mm:.2f} mm")
        kpi4.metric("Risk score", f"{float(df.loc[idx, 'risk_score']):.2f}")
        if ml_overlay is not None:
            kpi5.metric("ML band", str(ml_overlay.get("band", "n/a")))
            kpi6.metric("ML alert", "ON" if ml_overlay.get("alert_state", False) else "OFF")
            if fusion_state is not None:
                kpi7.metric("Fused alert", "ON" if fusion_state.get("fused_alert", False) else "OFF")
            else:
                kpi7.metric("Fused alert", "n/a")
        else:
            kpi5.metric("ML band", "n/a")
            kpi6.metric("ML alert", "n/a")
            kpi7.metric("Fused alert", "n/a")
    else:
        kpi2.metric("Fused settlement", f"{fused_mm:.2f} mm")
        kpi3.metric("Nearest InSAR obs", f"{nearest_insar_mm:.2f} mm")
        kpi4.metric("Modal frequency", f"{df.loc[idx, 'modal_freq_hz']:.3f} Hz")

    row1_left, row1_right = st.columns([1.05, 0.95])
    with row1_left:
        sinkhole_marker = (sinkhole_x_m, sinkhole_y_m) if mode == "Eisenhower Retrospective (InSAR-only)" else None
        st.plotly_chart(
            plan_view_fig(
                b,
                sinkhole_x_m,
                sinkhole_y_m,
                sigma_m,
                fused_mm,
                sensors,
                footprint_xy=footprint_xy,
                interp_grid=interp_grid,
                sinkhole_marker=sinkhole_marker,
                ml_overlay=ml_overlay,
            ),
            use_container_width=True,
        )
    with row1_right:
        st.plotly_chart(section_view_fig(b, sinkhole_x_m, sigma_m, fused_mm, section_profile=section_profile), use_container_width=True)

    projection_title = "3D Point Cloud Projection: Settlement from InSAR + Accelerometer Fusion"
    if mode == "Eisenhower Retrospective (InSAR-only)":
        if ml_overlay is None:
            projection_title = "3D Point Cloud Projection: Eisenhower InSAR-only Retrospective"
        else:
            projection_title = (
                "3D Point Cloud Projection: Eisenhower InSAR + ML Context "
                f"[{ml_overlay.get('band')} | {'ALERT' if ml_overlay.get('alert_state') else 'NO ALERT'} | "
                f"lead {ml_overlay.get('lead_days')} d]"
            )
    st.plotly_chart(point_cloud_fig(cloud, settlement_mm, ps, ps_settlement, sensors, title=projection_title), use_container_width=True)

    row2_left, row2_right = st.columns(2)
    if mode == "Eisenhower Retrospective (InSAR-only)":
        with row2_left:
            st.plotly_chart(insar_only_timeseries_fig(df, idx), use_container_width=True)
        with row2_right:
            st.plotly_chart(risk_score_fig(df, idx, retro_summary), use_container_width=True)
        if retro_summary:
            st.info(
                f"Event date: {retro_summary.get('event_date', '2023-08-16')}. "
                f"First alert date: {retro_summary.get('first_alert_date_in_claim_window')}. "
                f"Lead days: {retro_summary.get('lead_days_to_event')}."
            )
            if ml_overlay is not None:
                st.info(
                    f"ML conditioning: band {ml_overlay.get('band')} | "
                    f"state {'ALERT' if ml_overlay.get('alert_state', False) else 'NO ALERT'} | "
                    f"lead {ml_overlay.get('lead_days')} days."
                )
                if fusion_state is not None:
                    fused_score = fusion_state.get("fused_score")
                    fs = "n/a" if fused_score is None else f"{float(fused_score):.3f}"
                    st.info(
                        f"Fused rule {ml_overlay.get('fusion_rule')}: "
                        f"risk_on={fusion_state.get('risk_on')} | ml_on={fusion_state.get('ml_on')} | "
                        f"fused_score={fs} | final={'ALERT' if fusion_state.get('fused_alert') else 'NO ALERT'}."
                    )
            st.caption(
                "Geometry source: OpenStreetMap footprint for Eisenhower Parking Deck. InSAR source: "
                f"{retro_summary.get('source', 'unknown')}."
            )
            if sinkhole_xy_from_file is not None:
                st.caption("Sinkhole marker source: `sinkhole_location.csv` (user-provided lat/lon).")
            else:
                st.caption("Sinkhole marker is inferred from 'near the front of the deck' reports. Add `outputs/eisenhower_retrospective/sinkhole_location.csv` with `lat,lon` for surveyed location.")
    else:
        with row2_left:
            st.plotly_chart(time_series_fig(df, idx), use_container_width=True)
        with row2_right:
            st.plotly_chart(modal_fig(df, idx), use_container_width=True)

    st.markdown(
        """
**How to interpret for sinkhole risk**
- InSAR points give sparse absolute displacement anchors.
- Accelerometers provide continuous modal changes and dynamic behavior (synthetic mode).
- In the Eisenhower mode, risk score tracks displacement trend acceleration before August 16, 2023.
- The 3D projection maps fused settlement onto the building point cloud so localized bowl-shaped subsidence is visible in context.
"""
    )

    st.divider()
    st.header("Precursor Model Validation")
    available = {k: v for k, v in LOEO_RUN_DIRS.items() if v.exists()}
    if not available:
        st.info("No LOEO validation outputs found in configured output directories.")
        return

    run_name = st.selectbox("Validation run", options=list(available.keys()), index=0)
    bundle = load_loeo_bundle(available[run_name])
    if bundle is None:
        st.warning("Selected run does not contain recognized LOEO summary files.")
        return

    agg_df = bundle["aggregate"].copy()
    evt_df = bundle["events"].copy()
    evt_df["policy_f1"] = pd.to_numeric(evt_df.get("policy_f1"), errors="coerce")
    evt_df["policy_far_per_year"] = pd.to_numeric(evt_df.get("policy_far_per_year"), errors="coerce")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=agg_df["band"],
                y=agg_df["policy_f1_mean"],
                name="Mean F1",
                marker_color="#1f77b4",
            )
        )
        fig.update_layout(
            title="Band-Level Mean F1",
            xaxis_title="Band",
            yaxis_title="F1",
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=agg_df["band"],
                y=agg_df["policy_far_mean"],
                name="Mean FAR/year",
                marker_color="#d62728",
            )
        )
        fig.add_hline(y=3.0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Band-Level Mean False Alarms / Year",
            xaxis_title="Band",
            yaxis_title="FAR/year",
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    scatter = go.Figure()
    for band in sorted(evt_df["band"].dropna().unique()):
        b = evt_df[evt_df["band"] == band]
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
    st.plotly_chart(scatter, use_container_width=True)

    if bundle["benchmark"]:
        st.subheader("Frozen Eisenhower Benchmark")
        st.caption(bundle["benchmark"].get("frozen_benchmark_event", "unknown"))
        bench = pd.DataFrame(bundle["benchmark"].get("bands", []))
        if not bench.empty:
            cols = st.columns(min(3, len(bench)))
            for i, (_, r) in enumerate(bench.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(f"{r['band']} F1", f"{float(r['benchmark_policy_f1']):.3f}")
                    st.metric(f"{r['band']} FAR/year", f"{float(r['benchmark_policy_far_per_year']):.3f}")
                    st.metric(f"{r['band']} Lead (days)", f"{float(r['benchmark_policy_lead_days']):.1f}")

    st.subheader("Event Summary Table")
    show_cols = [
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
        if c in evt_df.columns
    ]
    st.dataframe(evt_df[show_cols].sort_values(["band", "holdout_event"]), use_container_width=True)


if __name__ == "__main__":
    main()
