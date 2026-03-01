#!/usr/bin/env python3
"""
Synthetic feasibility demo for a structural digital twin that fuses:
1) sparse low-frequency InSAR displacement observations, and
2) dense but drifting accelerometer-derived displacement proxies.

It also computes daily modal-frequency features from simulated acceleration
signals and runs anomaly detection for potential structural deterioration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal


@dataclass
class Config:
    seed: int = 7
    n_days: int = 180
    start_date: str = "2025-01-01"
    insar_interval_days: int = 6
    insar_noise_mm: float = 0.7
    accel_proxy_noise_mm: float = 2.0
    accel_proxy_drift_mm_per_day: float = 0.15
    damage_start_day: int = 95
    fs_hz: int = 100
    accel_window_sec: int = 180
    output_dir: str = "outputs"


def make_true_state(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    days = np.arange(cfg.n_days, dtype=float)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_days, freq="D")

    seasonal = 0.8 * np.sin(2 * np.pi * days / 45.0)
    creep = 0.03 * days
    post_damage = np.clip(days - cfg.damage_start_day, 0, None)
    settlement = 0.09 * post_damage + 0.002 * post_damage**1.2
    event = np.where(days > 130, 1.2 * np.exp(-(days - 130) / 12.0), 0.0)

    x_true = seasonal + creep + settlement + event
    x_true += rng.normal(0.0, 0.08, size=cfg.n_days).cumsum() * 0.03

    # Stiffness ratio k/k0 drops slowly after damage starts.
    k_ratio = np.ones(cfg.n_days)
    k_ratio[cfg.damage_start_day :] = np.clip(
        1.0 - 0.0025 * np.arange(cfg.n_days - cfg.damage_start_day), 0.65, 1.0
    )

    return pd.DataFrame({"date": dates, "day": days, "x_true_mm": x_true, "k_ratio": k_ratio})


def simulate_insar(cfg: Config, state: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    insar = pd.Series(np.nan, index=state.index, dtype=float)
    obs_idx = np.arange(0, cfg.n_days, cfg.insar_interval_days)
    keep = rng.random(obs_idx.size) > 0.1  # random acquisition gaps
    obs_idx = obs_idx[keep]
    insar.iloc[obs_idx] = (
        state.loc[obs_idx, "x_true_mm"].to_numpy()
        + rng.normal(0.0, cfg.insar_noise_mm, size=obs_idx.size)
    )
    return insar


def estimate_mode_frequency(accel_signal: np.ndarray, fs_hz: int) -> float:
    freqs, psd = signal.welch(accel_signal, fs=fs_hz, nperseg=min(len(accel_signal), 4096))
    band = (freqs > 0.5) & (freqs < 20.0)
    if not np.any(band):
        return np.nan
    bf = freqs[band]
    bp = psd[band]
    return float(bf[np.argmax(bp)])


def simulate_accel(cfg: Config, state: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    accel_proxy = np.zeros(cfg.n_days, dtype=float)
    modal_freq = np.zeros(cfg.n_days, dtype=float)

    drift = 0.0
    n = cfg.fs_hz * cfg.accel_window_sec
    t = np.arange(n) / cfg.fs_hz

    for i in range(cfg.n_days):
        # Dynamic frequencies tied to structural stiffness (f ~ sqrt(k)).
        k_ratio = state.loc[i, "k_ratio"]
        f1 = 5.4 * np.sqrt(k_ratio)
        f2 = 9.1 * np.sqrt(k_ratio)

        a = (
            0.06 * np.sin(2 * np.pi * f1 * t)
            + 0.03 * np.sin(2 * np.pi * f2 * t + 0.4)
            + rng.normal(0.0, 0.035, size=n)
        )
        modal_freq[i] = estimate_mode_frequency(a, cfg.fs_hz)

        drift += cfg.accel_proxy_drift_mm_per_day + rng.normal(0.0, 0.06)
        accel_proxy[i] = (
            state.loc[i, "x_true_mm"]
            + drift
            + rng.normal(0.0, cfg.accel_proxy_noise_mm)
        )

    return pd.Series(accel_proxy), pd.Series(modal_freq)


def kalman_fuse(
    accel_proxy: pd.Series,
    insar: pd.Series,
    q_pos: float = 0.08,
    q_vel: float = 0.02,
    r_accel: float = 7.0,
    r_insar: float = 0.6,
) -> pd.Series:
    n = len(accel_proxy)
    dt = 1.0
    A = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.diag([q_pos, q_vel])
    H = np.array([[1.0, 0.0]])
    x = np.array([accel_proxy.iloc[0], 0.0], dtype=float)
    P = np.diag([8.0, 1.5])
    out = np.zeros(n, dtype=float)

    for i in range(n):
        x = A @ x
        P = A @ P @ A.T + Q

        z_acc = accel_proxy.iloc[i]
        K = P @ H.T / (H @ P @ H.T + r_accel)
        x = x + (K.flatten() * (z_acc - (H @ x)[0]))
        P = (np.eye(2) - K @ H) @ P

        z_in = insar.iloc[i]
        if not np.isnan(z_in):
            K = P @ H.T / (H @ P @ H.T + r_insar)
            x = x + (K.flatten() * (z_in - (H @ x)[0]))
            P = (np.eye(2) - K @ H) @ P

        out[i] = x[0]

    return pd.Series(out)


def run_anomaly_detection(df: pd.DataFrame) -> pd.Series:
    baseline = df.iloc[:60]
    freq_mu = baseline["modal_freq_hz"].mean()
    freq_sigma = baseline["modal_freq_hz"].std(ddof=0) + 1e-6
    vel = df["fused_mm"].diff().fillna(0.0)
    vel_mu = baseline["fused_mm"].diff().fillna(0.0).mean()
    vel_sigma = baseline["fused_mm"].diff().fillna(0.0).std(ddof=0) + 1e-6

    freq_drop_z = (freq_mu - df["modal_freq_hz"]) / freq_sigma
    vel_z = np.abs((vel - vel_mu) / vel_sigma)
    anomaly = (freq_drop_z > 1.7) | (vel_z > 3.0)
    return anomaly


def rmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["x_true_mm"], label="True displacement (unknown in reality)", lw=2)
    ax.plot(df["date"], df["accel_proxy_mm"], label="Accel-only displacement proxy", alpha=0.6)
    ax.plot(df["date"], df["fused_mm"], label="Kalman fused estimate", lw=2.2)
    ax.scatter(df["date"], df["insar_mm"], s=18, label="InSAR observations", color="black", alpha=0.8)
    ax.set_title("Displacement Tracking: InSAR + Accelerometer Fusion")
    ax.set_ylabel("Displacement [mm]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "displacement_fusion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["modal_freq_hz"], label="Estimated fundamental mode", color="tab:blue")
    ax.axvline(df.loc[df["day"] == 95, "date"].iloc[0], color="tab:red", ls="--", label="Injected damage start")
    ax.scatter(
        df.loc[df["anomaly"], "date"],
        df.loc[df["anomaly"], "modal_freq_hz"],
        c="tab:red",
        s=16,
        label="Anomaly flags",
    )
    ax.set_title("Modal Frequency Drift and Anomaly Flags")
    ax.set_ylabel("Hz")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "modal_anomalies.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    state = make_true_state(cfg, rng)
    insar = simulate_insar(cfg, state, rng)
    accel_proxy, modal_freq = simulate_accel(cfg, state, rng)
    fused = kalman_fuse(accel_proxy, insar)

    df = state.copy()
    df["insar_mm"] = insar
    df["accel_proxy_mm"] = accel_proxy
    df["modal_freq_hz"] = modal_freq
    df["fused_mm"] = fused
    df["anomaly"] = run_anomaly_detection(df)

    metrics = {
        "rmse_accel_proxy_mm": rmse(df["x_true_mm"].to_numpy(), df["accel_proxy_mm"].to_numpy()),
        "rmse_fused_mm": rmse(df["x_true_mm"].to_numpy(), df["fused_mm"].to_numpy()),
        "mean_modal_freq_pre_damage_hz": float(df.loc[df["day"] < cfg.damage_start_day, "modal_freq_hz"].mean()),
        "mean_modal_freq_post_damage_hz": float(df.loc[df["day"] >= cfg.damage_start_day, "modal_freq_hz"].mean()),
        "anomaly_rate_pre_damage": float(df.loc[df["day"] < cfg.damage_start_day, "anomaly"].mean()),
        "anomaly_rate_post_damage": float(df.loc[df["day"] >= cfg.damage_start_day, "anomaly"].mean()),
        "insar_observation_count": int(df["insar_mm"].notna().sum()),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(outdir / "metrics.csv", index=False)
    df.to_csv(outdir / "synthetic_fusion_timeseries.csv", index=False)
    plot_results(df, outdir)

    print("Synthetic structural twin feasibility demo complete.")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
