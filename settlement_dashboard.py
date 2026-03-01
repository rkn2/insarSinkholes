#!/usr/bin/env python3
"""Streamlit dashboard for synthetic sinkhole/settlement structural twin monitoring."""

from __future__ import annotations

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
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def retro_paths() -> dict[str, Path]:
    candidates = [
        OUTPUTS_DIR / "eisenhower_retrospective_upgraded",
        OUTPUTS_DIR / "eisenhower_retrospective",
        Path("/Users/rebeccanapolitano/antigravityProjects/digitalTwins/structuralEx/outputs/eisenhower_retrospective"),
    ]
    selected = candidates[0]
    for d in candidates:
        if (d / "insar_retrospective_timeseries.csv").exists():
            selected = d
            break
    return {
        "retro_dir": selected,
        "retro_csv": selected / "insar_retrospective_timeseries.csv",
        "summary_json": selected / "retrospective_summary.json",
        "point_obs_csv": selected / "insar_point_observations.csv",
        "sinkhole_csv": selected / "sinkhole_location.csv",
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
    paths = retro_paths()
    if not paths["point_obs_csv"].exists():
        return None
    df = pd.read_csv(paths["point_obs_csv"])
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
    paths = retro_paths()
    if not paths["sinkhole_csv"].exists():
        return None
    df = pd.read_csv(paths["sinkhole_csv"])
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
        fig.add_trace(
            go.Scatter(
                x=[sinkhole_marker[0]],
                y=[sinkhole_marker[1]],
                mode="markers+text",
                text=["Reported sinkhole location"],
                textposition="top center",
                marker={"size": 14, "color": "magenta", "symbol": "x", "line": {"color": "white", "width": 1}},
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
    paths = retro_paths()
    if not paths["retro_csv"].exists():
        return None, {}
    df = pd.read_csv(paths["retro_csv"])
    if "date" not in df.columns:
        return None, {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    summary = {}
    if paths["summary_json"].exists():
        import json

        summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    return df, summary


def insar_only_timeseries_fig(df: pd.DataFrame, idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["displacement_mm"], mode="markers+lines", name="InSAR displacement (obs)", opacity=0.6))
    fig.add_trace(go.Scatter(x=df["date"], y=df["smoothed_mm"], mode="lines", name="Smoothed displacement", line={"width": 3}))
    if "changepoint" in df.columns:
        cp = df[df["changepoint"].astype(bool)]
        if len(cp):
            fig.add_trace(go.Scatter(x=cp["date"], y=cp["smoothed_mm"], mode="markers", name="Detected changepoint", marker={"color": "red", "size": 8}))
    if "slope_break_flag" in df.columns:
        sb = df[df["slope_break_flag"].astype(bool)]
        if len(sb):
            fig.add_trace(go.Scatter(x=sb["date"], y=sb["smoothed_mm"], mode="markers", name="Slope break", marker={"color": "magenta", "size": 9, "symbol": "x"}))
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
    if "accel_risk_z" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["accel_risk_z"],
                mode="lines",
                name="Acceleration risk z",
                line={"width": 2, "color": "mediumseagreen"},
                yaxis="y2",
            )
        )
    if "gaussian_bowl_risk" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["gaussian_bowl_risk"],
                mode="lines",
                name="Gaussian bowl risk",
                line={"width": 2, "color": "purple"},
                yaxis="y2",
            )
        )
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
        title="Precursor Risk Score + Component Signals (InSAR-only)",
        xaxis_title="Date",
        yaxis_title="score",
        yaxis2={"title": "component / mm/day", "overlaying": "y", "side": "right", "showgrid": False},
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
    )
    return fig


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

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Selected date", f"{df.loc[idx, 'date'].date()}")
    if mode == "Eisenhower Retrospective (InSAR-only)":
        kpi2.metric("Smoothed InSAR", f"{fused_mm:.2f} mm")
        kpi3.metric("Observed InSAR", f"{nearest_insar_mm:.2f} mm")
        kpi4.metric("Risk score", f"{float(df.loc[idx, 'risk_score']):.2f}")
        extra1, extra2, extra3 = st.columns(3)
        extra1.metric("Accel risk z", f"{float(df.loc[idx, 'accel_risk_z']):.2f}" if "accel_risk_z" in df.columns else "n/a")
        extra2.metric("Gaussian bowl (mm)", f"{float(df.loc[idx, 'gaussian_bowl_mm']):.1f}" if "gaussian_bowl_mm" in df.columns and pd.notna(df.loc[idx, 'gaussian_bowl_mm']) else "n/a")
        extra3.metric("Gaussian fit R2", f"{float(df.loc[idx, 'gaussian_fit_r2']):.2f}" if "gaussian_fit_r2" in df.columns and pd.notna(df.loc[idx, 'gaussian_fit_r2']) else "n/a")
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
            ),
            use_container_width=True,
        )
    with row1_right:
        st.plotly_chart(section_view_fig(b, sinkhole_x_m, sigma_m, fused_mm, section_profile=section_profile), use_container_width=True)

    projection_title = (
        "3D Point Cloud Projection: Eisenhower InSAR-only Retrospective"
        if mode == "Eisenhower Retrospective (InSAR-only)"
        else "3D Point Cloud Projection: Settlement from InSAR + Accelerometer Fusion"
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
            if retro_summary.get("slope_break_date"):
                st.caption(
                    f"Slope break date: {retro_summary.get('slope_break_date')} "
                    f"(p={retro_summary.get('slope_break_p_value')})."
                )
            if retro_summary.get("data_discovery_warning"):
                st.caption(f"Data discovery warning: {retro_summary.get('data_discovery_warning')}")
            paths = retro_paths()
            st.caption(f"Retrospective input directory: `{paths['retro_dir']}`")
            st.caption(
                "Geometry source: OpenStreetMap footprint for Eisenhower Parking Deck. InSAR source: "
                f"{retro_summary.get('source', 'unknown')}."
            )
            if sinkhole_xy_from_file is not None:
                st.caption("Sinkhole marker source: `sinkhole_location.csv` (user-provided lat/lon).")
            else:
                st.caption("Sinkhole marker is inferred from 'near the front of the deck' reports. Add `sinkhole_location.csv` with `lat,lon` in the active retrospective outputs directory.")
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


if __name__ == "__main__":
    main()
