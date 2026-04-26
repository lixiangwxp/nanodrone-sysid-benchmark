import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.model_comparison import require_csv
from utils.plot_utils import setup_matplotlib


ROT_STATES = ["rx", "ry", "rz"]
OMEGA_STATES = ["wx", "wy", "wz"]
MOTOR_COLUMNS = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]
MODE_COLUMNS = ["T", "tau_x", "tau_y", "tau_z"]
REGIME_COLUMNS = ["tilt_deg", "omega_norm", "motor_change_rate", "speed_norm"]

ARM_LENGTH = 0.0353
KT = 3.72e-08
KC = 7.74e-12
MASS = 0.045
GRAVITY = 9.81
THRUST_TO_WEIGHT = 2.0
T_MAX = THRUST_TO_WEIGHT * MASS * GRAVITY
MAX_TORQUE = np.array([1e-2, 1e-2, 3e-3], dtype=float)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare base oracle residual distributions and raw trajectory "
            "regimes across trajectory families."
        )
    )
    parser.add_argument(
        "--prediction-csv",
        action="append",
        dest="prediction_csvs",
        default=[],
        help="Baseline prediction CSV. Repeat once per trajectory label.",
    )
    parser.add_argument(
        "--traj-label",
        action="append",
        dest="traj_labels",
        default=[],
        help="Trajectory label for the corresponding --prediction-csv.",
    )
    parser.add_argument(
        "--raw-traj",
        action="append",
        dest="raw_trajs",
        default=[],
        help=(
            "Raw trajectory files as LABEL:GLOB. Repeat for random/square/chirp/melon. "
            "Quote the argument so the shell does not expand '*'."
        ),
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--plot-dir", required=True, type=str)
    parser.add_argument(
        "--run-tsne",
        action="store_true",
        help="Also save residual_tsne_by_traj.png. PCA is always generated.",
    )
    parser.add_argument(
        "--max-tsne-samples",
        type=int,
        default=3000,
        help="Maximum residual samples used for t-SNE.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional CSV path for distribution summary statistics.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not args.prediction_csvs:
        raise ValueError("At least one --prediction-csv is required.")
    if len(args.prediction_csvs) != len(args.traj_labels):
        raise ValueError("--traj-label count must match --prediction-csv count.")
    if not args.raw_trajs:
        raise ValueError("At least one --raw-traj LABEL:GLOB is required.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be > 0.")
    if args.max_tsne_samples <= 1:
        raise ValueError("--max-tsne-samples must be > 1.")


def require_columns(df, columns, source):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def parse_raw_traj_specs(raw_trajs):
    specs = []
    for spec in raw_trajs:
        parts = spec.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"--raw-traj must be LABEL:GLOB, got {spec!r}")
        label, pattern = parts
        files = sorted(Path(path).resolve() for path in glob.glob(os.path.expanduser(pattern)))
        if not files:
            raise FileNotFoundError(f"No raw files matched {pattern!r} for label {label!r}")
        specs.append((label, files))
    return specs


def estimate_dt(df):
    if "t" not in df.columns or len(df) < 2:
        return 0.01
    dt_values = np.diff(df["t"].to_numpy(float))
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if dt_values.size == 0:
        return 0.01
    return float(np.median(dt_values))


def motor_to_phys_np(motors):
    motors = np.asarray(motors, dtype=float)
    omega2 = motors ** 2
    thrust = KT * omega2.sum(axis=1)
    tau_x = KT * ARM_LENGTH * (
        (omega2[:, 2] + omega2[:, 3]) - (omega2[:, 0] + omega2[:, 1])
    )
    tau_y = KT * ARM_LENGTH * (
        (omega2[:, 1] + omega2[:, 2]) - (omega2[:, 0] + omega2[:, 3])
    )
    tau_z = KC * ((omega2[:, 0] + omega2[:, 2]) - (omega2[:, 1] + omega2[:, 3]))
    return np.column_stack(
        [thrust / T_MAX, tau_x / MAX_TORQUE[0], tau_y / MAX_TORQUE[1], tau_z / MAX_TORQUE[2]]
    )


def quat_to_tilt_deg(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)
    valid = np.isfinite(quat_xyzw).all(axis=1)
    tilt = np.full(len(quat_xyzw), np.nan, dtype=float)
    if not valid.any():
        return tilt
    body_z_world = R.from_quat(quat_xyzw[valid]).apply(
        np.tile(np.array([[0.0, 0.0, 1.0]]), (valid.sum(), 1))
    )
    z_component = np.clip(body_z_world[:, 2], -1.0, 1.0)
    tilt[valid] = np.degrees(np.arccos(z_component))
    return tilt


def build_residual_frame(csv_path, label, horizon):
    df = require_csv(Path(csv_path).expanduser().resolve())
    pred_cols = [f"{state}_pred_h{horizon}" for state in [*ROT_STATES, *OMEGA_STATES]]
    require_columns(df, ["t", *ROT_STATES, *OMEGA_STATES, *pred_cols], str(csv_path))

    max_start = len(df) - horizon + 1
    if max_start <= 0:
        raise ValueError(f"{csv_path} is too short for horizon {horizon}.")

    start_time = df["t"].iloc[:max_start].to_numpy(float)
    true_rot = df[ROT_STATES].shift(-(horizon - 1)).iloc[:max_start].to_numpy(float)
    pred_rot = df[[f"{state}_pred_h{horizon}" for state in ROT_STATES]].iloc[
        :max_start
    ].to_numpy(float)
    true_omega = df[OMEGA_STATES].shift(-(horizon - 1)).iloc[:max_start].to_numpy(float)
    pred_omega = df[[f"{state}_pred_h{horizon}" for state in OMEGA_STATES]].iloc[
        :max_start
    ].to_numpy(float)

    finite = (
        np.isfinite(start_time)
        & np.isfinite(true_rot).all(axis=1)
        & np.isfinite(pred_rot).all(axis=1)
        & np.isfinite(true_omega).all(axis=1)
        & np.isfinite(pred_omega).all(axis=1)
    )
    rot_residual = true_rot[finite] - pred_rot[finite]
    omega_residual = true_omega[finite] - pred_omega[finite]
    frame = pd.DataFrame(
        {
            "traj_label": label,
            "start_time": start_time[finite],
            "rot_residual_rx": rot_residual[:, 0],
            "rot_residual_ry": rot_residual[:, 1],
            "rot_residual_rz": rot_residual[:, 2],
            "omega_residual_wx": omega_residual[:, 0],
            "omega_residual_wy": omega_residual[:, 1],
            "omega_residual_wz": omega_residual[:, 2],
            "rot_residual_norm": np.linalg.norm(rot_residual, axis=1),
            "omega_residual_norm": np.linalg.norm(omega_residual, axis=1),
        }
    )
    frame["horizon"] = horizon
    return frame


def build_raw_regime_frame(files, label):
    rows = []
    for file_path in files:
        df = pd.read_csv(file_path)
        require_columns(
            df,
            [
                "t",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "qx",
                "qy",
                "qz",
                "qw",
                "wx",
                "wy",
                "wz",
                *MOTOR_COLUMNS,
            ],
            str(file_path),
        )
        dt = estimate_dt(df)
        motors = df[MOTOR_COLUMNS].to_numpy(float)
        mode = motor_to_phys_np(motors)
        motor_delta = np.vstack([np.zeros((1, motors.shape[1])), np.diff(motors, axis=0)])
        motor_change_rate = np.linalg.norm(motor_delta, axis=1) / max(dt, 1e-12)
        omega = df[OMEGA_STATES].to_numpy(float)
        vel = df[["vx", "vy", "vz"]].to_numpy(float)

        frame = pd.DataFrame(
            {
                "traj_label": label,
                "source_file": str(file_path),
                "t": df["t"].to_numpy(float),
                "tilt_deg": quat_to_tilt_deg(df[["qx", "qy", "qz", "qw"]].to_numpy(float)),
                "omega_norm": np.linalg.norm(omega, axis=1),
                "motor_change_rate": motor_change_rate,
                "speed_norm": np.linalg.norm(vel, axis=1),
            }
        )
        for idx, column in enumerate(MODE_COLUMNS):
            frame[f"mode_{column}"] = mode[:, idx]
        rows.append(frame.replace([np.inf, -np.inf], np.nan).dropna())

    if not rows:
        raise ValueError(f"No raw regime rows built for label {label!r}.")
    return pd.concat(rows, ignore_index=True)


def plot_residual_distributions(residual_df, plot_dir):
    setup_matplotlib()
    labels = list(residual_df["traj_label"].drop_duplicates())
    colors = plt.get_cmap("tab10").colors
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), dpi=180)
    specs = [
        ("rot_residual_norm", "Rot oracle residual norm [rad]"),
        ("omega_residual_norm", "Omega oracle residual norm [rad/s]"),
    ]
    for ax, (column, xlabel) in zip(axs, specs):
        for idx, label in enumerate(labels):
            values = residual_df.loc[residual_df["traj_label"] == label, column].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            color = colors[idx % len(colors)]
            if is_effectively_constant(values):
                ax.axvline(values[0], linewidth=1.5, label=label, color=color)
            else:
                ax.hist(
                    values,
                    bins=60,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=label,
                    color=color,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    axs[0].legend()
    fig.tight_layout()

    plot_path = plot_dir / "residual_distribution_by_traj.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_embedding(residual_df, plot_dir, method="pca", max_samples=None):
    residual_columns = [
        "rot_residual_rx",
        "rot_residual_ry",
        "rot_residual_rz",
        "omega_residual_wx",
        "omega_residual_wy",
        "omega_residual_wz",
    ]
    data = residual_df[["traj_label", *residual_columns]].replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    if max_samples is not None and len(data) > max_samples:
        data = data.sample(max_samples, random_state=0)
    if len(data) < 3:
        raise ValueError("Need at least 3 residual samples for embedding plots.")

    x = StandardScaler().fit_transform(data[residual_columns].to_numpy(float))
    if method == "pca":
        embedding = PCA(n_components=2, random_state=0).fit_transform(x)
        filename = "residual_pca_by_traj.png"
        xlabel, ylabel = "PC1", "PC2"
    elif method == "tsne":
        perplexity = min(30, max(2, (len(data) - 1) // 3))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=0,
        ).fit_transform(x)
        filename = "residual_tsne_by_traj.png"
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
    else:
        raise ValueError(f"Unknown embedding method {method!r}.")

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=180)
    labels = list(data["traj_label"].drop_duplicates())
    colors = plt.get_cmap("tab10").colors
    for idx, label in enumerate(labels):
        mask = data["traj_label"].to_numpy() == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=6,
            alpha=0.35,
            label=label,
            color=colors[idx % len(colors)],
            edgecolors="none",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=2.0)
    fig.tight_layout()

    plot_path = plot_dir / filename
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_raw_distributions(raw_df, columns, xlabels, filename, plot_dir):
    setup_matplotlib()
    labels = list(raw_df["traj_label"].drop_duplicates())
    colors = plt.get_cmap("tab10").colors
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=180)
    for ax, column, xlabel in zip(axs.flat, columns, xlabels):
        for idx, label in enumerate(labels):
            values = raw_df.loc[raw_df["traj_label"] == label, column].to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            color = colors[idx % len(colors)]
            if is_effectively_constant(values):
                ax.axvline(values[0], linewidth=1.4, label=label, color=color)
            else:
                ax.hist(
                    values,
                    bins=60,
                    density=True,
                    histtype="step",
                    linewidth=1.4,
                    label=label,
                    color=color,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    axs.flat[0].legend()
    fig.tight_layout()

    plot_path = plot_dir / filename
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def build_summary(residual_df, raw_df):
    rows = []
    for label in sorted(set(residual_df["traj_label"]).union(raw_df["traj_label"])):
        row = {"traj_label": label}
        r_group = residual_df[residual_df["traj_label"] == label]
        raw_group = raw_df[raw_df["traj_label"] == label]
        for column in ["rot_residual_norm", "omega_residual_norm"]:
            values = r_group[column].to_numpy(float) if column in r_group else np.array([])
            row[f"{column}_mean"] = float(np.nanmean(values)) if values.size else np.nan
            row[f"{column}_p95"] = float(np.nanpercentile(values, 95)) if values.size else np.nan
        for column in [*REGIME_COLUMNS, *[f"mode_{column}" for column in MODE_COLUMNS]]:
            values = raw_group[column].to_numpy(float) if column in raw_group else np.array([])
            row[f"{column}_mean"] = float(np.nanmean(values)) if values.size else np.nan
            row[f"{column}_p95"] = float(np.nanpercentile(values, 95)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def is_effectively_constant(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return True
    spread = float(np.nanmax(values) - np.nanmin(values))
    scale = max(float(np.nanmax(np.abs(values))), 1.0)
    return spread <= 1e-12 * scale


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    plot_dir = Path(args.plot_dir).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    residual_df = pd.concat(
        [
            build_residual_frame(csv_path, label, args.horizon)
            for csv_path, label in zip(args.prediction_csvs, args.traj_labels)
        ],
        ignore_index=True,
    )
    raw_df = pd.concat(
        [build_raw_regime_frame(files, label) for label, files in parse_raw_traj_specs(args.raw_trajs)],
        ignore_index=True,
    )

    paths = [
        plot_residual_distributions(residual_df, plot_dir),
        plot_embedding(residual_df, plot_dir, method="pca"),
        plot_raw_distributions(
            raw_df,
            [f"mode_{column}" for column in MODE_COLUMNS],
            ["T norm", "tau_x norm", "tau_y norm", "tau_z norm"],
            "control_mode_distribution_by_traj.png",
            plot_dir,
        ),
        plot_raw_distributions(
            raw_df,
            REGIME_COLUMNS,
            ["Tilt [deg]", "Omega norm [rad/s]", "Motor change rate [rad/s^2]", "Speed norm [m/s]"],
            "state_regime_distribution_by_traj.png",
            plot_dir,
        ),
    ]
    if args.run_tsne:
        paths.append(
            plot_embedding(
                residual_df,
                plot_dir,
                method="tsne",
                max_samples=args.max_tsne_samples,
            )
        )

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path is not None
        else plot_dir / "residual_regime_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    build_summary(residual_df, raw_df).to_csv(summary_path, index=False)
    residual_df.to_csv(plot_dir / "residual_samples.csv", index=False)
    raw_df.to_csv(plot_dir / "raw_regime_samples.csv", index=False)

    for path in paths:
        print(f"Saved plot to: {path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved residual samples to: {plot_dir / 'residual_samples.csv'}")
    print(f"Saved raw regime samples to: {plot_dir / 'raw_regime_samples.csv'}")


if __name__ == "__main__":
    main()
