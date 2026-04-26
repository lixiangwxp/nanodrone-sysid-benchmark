import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.model_comparison import add_rotation_columns, require_csv
from utils.plot_utils import setup_matplotlib


ROT_STATES = ["rx", "ry", "rz"]
OMEGA_STATES = ["wx", "wy", "wz"]
QUAT_STATES = ["qx", "qy", "qz", "qw"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot oracle rot/omega residuals and mean horizon errors over "
            "rollout start time for one baseline prediction CSV."
        )
    )
    parser.add_argument("--prediction-csv", required=True, type=str)
    parser.add_argument("--model-label", default="base", type=str)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--plot-dir", required=True, type=str)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Rolling mean window in samples. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional CSV path for the time-aligned diagnostic values.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0.")
    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be > 0.")
    if args.horizon > args.max_horizon:
        raise ValueError("--horizon must be <= --max-horizon.")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be > 0.")


def require_columns(df, columns, source):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def estimate_dt(df):
    if "t" not in df.columns or len(df) < 2:
        return 0.01
    dt_values = np.diff(df["t"].to_numpy(float))
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if dt_values.size == 0:
        return 0.01
    return float(np.median(dt_values))


def quat_geodesic_distance(q_true, q_pred):
    q_true = np.asarray(q_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    q_true_norm = np.linalg.norm(q_true, axis=1, keepdims=True)
    q_pred_norm = np.linalg.norm(q_pred, axis=1, keepdims=True)
    valid = (
        np.isfinite(q_true).all(axis=1)
        & np.isfinite(q_pred).all(axis=1)
        & (q_true_norm[:, 0] > 0)
        & (q_pred_norm[:, 0] > 0)
    )
    out = np.full(q_true.shape[0], np.nan, dtype=float)
    if not valid.any():
        return out
    q_true_valid = q_true[valid] / q_true_norm[valid]
    q_pred_valid = q_pred[valid] / q_pred_norm[valid]
    dot = np.abs(np.sum(q_true_valid * q_pred_valid, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    out[valid] = 2.0 * np.arccos(dot)
    return out


def aligned_state_arrays(df, states, horizon):
    pred_cols = [f"{state}_pred_h{horizon}" for state in states]
    require_columns(df, ["t", *states, *pred_cols], "prediction CSV")

    max_start = len(df) - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Prediction CSV has {len(df)} rows, too short for horizon {horizon}."
        )

    start_time = df["t"].iloc[:max_start].to_numpy(float)
    true_values = df[states].shift(-(horizon - 1)).iloc[:max_start].to_numpy(float)
    pred_values = df[pred_cols].iloc[:max_start].to_numpy(float)
    mask = (
        np.isfinite(start_time)
        & np.isfinite(true_values).all(axis=1)
        & np.isfinite(pred_values).all(axis=1)
    )
    return start_time[mask], true_values[mask], pred_values[mask], mask


def compute_mean_errors_by_start_time(df, max_horizon):
    n_common = len(df) - max_horizon + 1
    if n_common <= 0:
        raise ValueError(
            f"Prediction CSV has {len(df)} rows, too short for max horizon {max_horizon}."
        )

    start_time = df["t"].iloc[:n_common].to_numpy(float)
    rot_errors = []
    omega_errors = []

    for horizon in range(1, max_horizon + 1):
        omega_pred_cols = [f"{state}_pred_h{horizon}" for state in OMEGA_STATES]
        quat_pred_cols = [f"{state}_pred_h{horizon}" for state in QUAT_STATES]
        require_columns(
            df,
            [*OMEGA_STATES, *omega_pred_cols, *QUAT_STATES, *quat_pred_cols],
            "prediction CSV",
        )

        true_omega = (
            df[OMEGA_STATES].shift(-(horizon - 1)).iloc[:n_common].to_numpy(float)
        )
        pred_omega = df[omega_pred_cols].iloc[:n_common].to_numpy(float)
        omega_errors.append(np.linalg.norm(true_omega - pred_omega, axis=1))

        true_quat = (
            df[QUAT_STATES].shift(-(horizon - 1)).iloc[:n_common].to_numpy(float)
        )
        pred_quat = df[quat_pred_cols].iloc[:n_common].to_numpy(float)
        rot_errors.append(quat_geodesic_distance(true_quat, pred_quat))

    rot_error_mean = np.nanmean(np.vstack(rot_errors), axis=0)
    omega_error_mean = np.nanmean(np.vstack(omega_errors), axis=0)
    finite = (
        np.isfinite(start_time)
        & np.isfinite(rot_error_mean)
        & np.isfinite(omega_error_mean)
    )
    return pd.DataFrame(
        {
            "start_time": start_time[finite],
            "mean_rot_error_h1_to_hmax": rot_error_mean[finite],
            "mean_omega_error_h1_to_hmax": omega_error_mean[finite],
        }
    )


def build_summary(df, horizon, max_horizon):
    df = add_rotation_columns(df)
    dt = estimate_dt(df)

    start_time, true_rot, pred_rot, rot_mask = aligned_state_arrays(
        df, ROT_STATES, horizon
    )
    start_time_omega, true_omega, pred_omega, omega_mask = aligned_state_arrays(
        df, OMEGA_STATES, horizon
    )
    if len(start_time) != len(start_time_omega) or not np.allclose(
        start_time, start_time_omega
    ):
        raise ValueError("Rot and omega residual rows did not align.")

    rot_residual = true_rot - pred_rot
    omega_residual = true_omega - pred_omega
    target_time = start_time + float(horizon) * dt
    residual_df = pd.DataFrame(
        {
            "start_time": start_time,
            "target_time": target_time,
            "rot_residual_rx_h": rot_residual[:, 0],
            "rot_residual_ry_h": rot_residual[:, 1],
            "rot_residual_rz_h": rot_residual[:, 2],
            "omega_residual_wx_h": omega_residual[:, 0],
            "omega_residual_wy_h": omega_residual[:, 1],
            "omega_residual_wz_h": omega_residual[:, 2],
            "rot_residual_norm_h": np.linalg.norm(rot_residual, axis=1),
            "omega_residual_norm_h": np.linalg.norm(omega_residual, axis=1),
        }
    )

    mean_df = compute_mean_errors_by_start_time(df, max_horizon)
    summary_df = residual_df.merge(mean_df, on="start_time", how="left")
    summary_df["horizon"] = horizon
    summary_df["max_horizon"] = max_horizon
    summary_df["dt_estimate"] = dt
    return summary_df


def rolling_mean(values, window):
    series = pd.Series(values)
    return series.rolling(window, center=True, min_periods=1).mean().to_numpy()


def plot_residual_components(summary_df, model_label, horizon, plot_dir, smooth_window):
    setup_matplotlib()
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=180)
    x = summary_df["start_time"].to_numpy(float)
    groups = [
        ("Rot residual", ["rx", "ry", "rz"], "rot_residual_{}_h", "rad"),
        ("Omega residual", ["wx", "wy", "wz"], "omega_residual_{}_h", "rad/s"),
    ]

    for ax, (title, labels, template, unit) in zip(axs, groups):
        for label in labels:
            y = summary_df[template.format(label)].to_numpy(float)
            ax.plot(x, y, alpha=0.25, linewidth=0.8)
            if smooth_window > 1:
                ax.plot(
                    x,
                    rolling_mean(y, smooth_window),
                    linewidth=1.4,
                    label=f"{label} smooth",
                )
            else:
                ax.plot(x, y, linewidth=1.2, label=label)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_title(f"{title}, h={horizon}, {model_label}")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Rollout start time [s]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"oracle_residual_h{horizon}_vs_time.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_mean_errors(summary_df, model_label, max_horizon, plot_dir, smooth_window):
    setup_matplotlib()
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=180)
    x = summary_df["start_time"].to_numpy(float)
    specs = [
        ("mean_rot_error_h1_to_hmax", "Mean rot error h=1..H", "rad"),
        ("mean_omega_error_h1_to_hmax", "Mean omega error h=1..H", "rad/s"),
    ]

    for ax, (column, title, unit) in zip(axs, specs):
        y = summary_df[column].to_numpy(float)
        ax.plot(x, y, color="0.45", alpha=0.35, linewidth=0.8, label="raw")
        if smooth_window > 1:
            ax.plot(
                x,
                rolling_mean(y, smooth_window),
                color="tab:blue",
                linewidth=1.6,
                label="smooth",
            )
        ax.set_title(f"{title}, H={max_horizon}, {model_label}")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axs[-1].set_xlabel("Rollout start time [s]")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "mean_rotomega_error_vs_time.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    prediction_path = Path(args.prediction_csv).expanduser().resolve()
    plot_dir = Path(args.plot_dir).expanduser().resolve()
    df = require_csv(prediction_path)
    summary_df = build_summary(df, args.horizon, args.max_horizon)

    residual_path = plot_residual_components(
        summary_df,
        args.model_label,
        args.horizon,
        plot_dir,
        args.smooth_window,
    )
    mean_error_path = plot_mean_errors(
        summary_df,
        args.model_label,
        args.max_horizon,
        plot_dir,
        args.smooth_window,
    )

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path is not None
        else plot_dir / "oracle_residual_time_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved residual plot to: {residual_path}")
    print(f"Saved mean-error plot to: {mean_error_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
