import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.latex_utils import print_latex_table_results
from utils.metrics_utils import compute_errors, compute_simerr
from utils.plot_utils import setup_matplotlib


SUMMARY_COLUMNS = [
    "created_at",
    "model_label",
    "model_family",
    "loss_name",
    "model",
    "pos_h1",
    "pos_h10",
    "pos_h50",
    "vel_h1",
    "vel_h10",
    "vel_h50",
    "rot_h1",
    "rot_h10",
    "rot_h50",
    "omega_h1",
    "omega_h10",
    "omega_h50",
    "sim_pos",
    "sim_vel",
    "sim_rot",
    "sim_omega",
]


def so3_log_to_quat_np(r):
    rot = R.from_rotvec(r)
    return rot.as_quat()


def quat_to_euler_np(q_xyzw):
    rot = R.from_quat(q_xyzw)
    return rot.as_euler("xyz", degrees=False)


def add_rotation_columns(df):
    df = df.copy()
    new_cols = {}

    rx_cols = [c for c in df.columns if c.startswith("rx")]

    for rx_col in rx_cols:
        suffix = rx_col[2:]
        ry_col = f"ry{suffix}"
        rz_col = f"rz{suffix}"

        if ry_col not in df.columns or rz_col not in df.columns:
            continue

        r = df[[rx_col, ry_col, rz_col]].to_numpy(float)
        valid_mask = np.isfinite(r).all(axis=1)
        q = np.full((len(df), 4), np.nan, dtype=float)
        e = np.full((len(df), 3), np.nan, dtype=float)

        if valid_mask.any():
            q_valid = so3_log_to_quat_np(r[valid_mask])
            e_valid = quat_to_euler_np(q_valid)
            q[valid_mask] = q_valid
            e[valid_mask] = e_valid

        new_cols[f"roll{suffix}"] = e[:, 0]
        new_cols[f"pitch{suffix}"] = e[:, 1]
        new_cols[f"yaw{suffix}"] = e[:, 2]
        new_cols[f"roll{suffix}_deg"] = np.degrees(e[:, 0])
        new_cols[f"pitch{suffix}_deg"] = np.degrees(e[:, 1])
        new_cols[f"yaw{suffix}_deg"] = np.degrees(e[:, 2])
        new_cols[f"qx{suffix}"] = q[:, 0]
        new_cols[f"qy{suffix}"] = q[:, 1]
        new_cols[f"qz{suffix}"] = q[:, 2]
        new_cols[f"qw{suffix}"] = q[:, 3]

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing prediction file: {path}")
    return pd.read_csv(path)


def build_summary_row(created_at, model_label, model_family, loss_name, metrics):
    sim_p, sim_v, sim_r, sim_w = compute_simerr(metrics)
    return {
        "created_at": created_at,
        "model_label": model_label,
        "model_family": model_family,
        "loss_name": loss_name,
        "model": model_label,
        "pos_h1": metrics["pos"][1],
        "pos_h10": metrics["pos"][10],
        "pos_h50": metrics["pos"][50],
        "vel_h1": metrics["vel"][1],
        "vel_h10": metrics["vel"][10],
        "vel_h50": metrics["vel"][50],
        "rot_h1": metrics["rot"][1],
        "rot_h10": metrics["rot"][10],
        "rot_h50": metrics["rot"][50],
        "omega_h1": metrics["omega"][1],
        "omega_h10": metrics["omega"][10],
        "omega_h50": metrics["omega"][50],
        "sim_pos": sim_p,
        "sim_vel": sim_v,
        "sim_rot": sim_r,
        "sim_omega": sim_w,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics from one or more explicit prediction CSVs"
    )
    parser.add_argument(
        "--prediction-csv",
        action="append",
        dest="prediction_csvs",
        default=[],
        help="Prediction CSV path. Repeat for multiple experiments.",
    )
    parser.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        default=[],
        help="Display name for each prediction CSV. Repeat in the same order as --prediction-csv.",
    )
    parser.add_argument(
        "--model-family",
        action="append",
        dest="model_families",
        default=[],
        help="Optional model family for each prediction CSV. Repeat in the same order as --prediction-csv.",
    )
    parser.add_argument(
        "--loss-name",
        action="append",
        dest="loss_names",
        default=[],
        help="Optional loss name for each prediction CSV. Repeat in the same order as --prediction-csv.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=50,
        help="Maximum prediction horizon to evaluate.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Render metric and trajectory comparison plots for the supplied experiments.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(PROJECT_ROOT / "out" / "benchmark_summary.csv"),
        help="CSV file to append the computed metrics to.",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    experiment_count = len(args.prediction_csvs)
    if experiment_count == 0:
        raise ValueError("At least one --prediction-csv must be provided.")

    if len(args.model_labels) != experiment_count:
        raise ValueError(
            "--model-label count must match --prediction-csv count "
            f"(got {len(args.model_labels)} labels for {experiment_count} CSVs)."
        )

    if args.model_families and len(args.model_families) != experiment_count:
        raise ValueError(
            "--model-family count must match --prediction-csv count when provided "
            f"(got {len(args.model_families)} families for {experiment_count} CSVs)."
        )

    if args.loss_names and len(args.loss_names) != experiment_count:
        raise ValueError(
            "--loss-name count must match --prediction-csv count when provided "
            f"(got {len(args.loss_names)} values for {experiment_count} CSVs)."
        )

    if args.max_horizon < 50:
        raise ValueError("--max-horizon must be at least 50 for the fixed *_h50 summary columns")


def build_experiments(args):
    experiment_count = len(args.prediction_csvs)
    model_families = args.model_families or [""] * experiment_count
    loss_names = args.loss_names or [""] * experiment_count

    experiments = []
    for idx in range(experiment_count):
        experiments.append(
            {
                "prediction_csv": Path(args.prediction_csvs[idx]).expanduser().resolve(),
                "model_label": args.model_labels[idx],
                "model_family": model_families[idx],
                "loss_name": loss_names[idx],
            }
        )
    return experiments


def evaluate_experiments(experiments, max_horizon):
    evaluated = []
    for experiment in experiments:
        df_pred = add_rotation_columns(require_csv(experiment["prediction_csv"]))
        metrics = compute_errors(df_pred, max_horizon)
        evaluated.append(
            {
                **experiment,
                "df_pred": df_pred,
                "metrics": metrics,
            }
        )
    return evaluated


def build_summary_df(evaluated_experiments, created_at):
    rows = [
        build_summary_row(
            created_at=created_at,
            model_label=experiment["model_label"],
            model_family=experiment["model_family"],
            loss_name=experiment["loss_name"],
            metrics=experiment["metrics"],
        )
        for experiment in evaluated_experiments
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def normalize_existing_summary(existing_df):
    normalized = existing_df.copy()

    if "model_label" not in normalized.columns:
        if "model" in normalized.columns:
            normalized["model_label"] = normalized["model"]
        else:
            normalized["model_label"] = ""
    if "model" not in normalized.columns:
        normalized["model"] = normalized["model_label"]
    if "model_family" not in normalized.columns:
        normalized["model_family"] = ""
    if "loss_name" not in normalized.columns:
        normalized["loss_name"] = ""
    if "created_at" not in normalized.columns:
        normalized["created_at"] = ""

    for column in SUMMARY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan

    return normalized[SUMMARY_COLUMNS]


def load_existing_summary(summary_path):
    if not summary_path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    try:
        existing_df = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    return normalize_existing_summary(existing_df)


def append_summary(summary_path, new_rows_df):
    existing_df = load_existing_summary(summary_path)
    if existing_df.empty:
        combined_df = new_rows_df.copy()
    else:
        combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
    combined_df = combined_df[SUMMARY_COLUMNS]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_path.with_name(f"{summary_path.name}.tmp")
    combined_df.to_csv(tmp_path, index=False)
    tmp_path.replace(summary_path)
    return combined_df


def print_latex_summary(evaluated_experiments, max_horizon):
    target_horizons = [1, 10, min(50, max_horizon)]
    latex_rows = []

    for experiment in evaluated_experiments:
        metrics = experiment["metrics"]
        sim_p, sim_v, sim_r, sim_w = compute_simerr(metrics)
        latex_rows.append(
            [
                experiment["model_label"],
                metrics["pos"][1],
                metrics["pos"][10],
                metrics["pos"][target_horizons[-1]],
                sim_p,
                metrics["vel"][1],
                metrics["vel"][10],
                metrics["vel"][target_horizons[-1]],
                sim_v,
                metrics["rot"][1],
                metrics["rot"][10],
                metrics["rot"][target_horizons[-1]],
                sim_r,
                metrics["omega"][1],
                metrics["omega"][10],
                metrics["omega"][target_horizons[-1]],
                sim_w,
            ]
        )

    print_latex_table_results(latex_rows, target_horizons)


def show_metric_plots(evaluated_experiments):
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.5), sharex=True)
    metric_names = ["pos", "vel", "rot", "omega"]
    ylabels = [
        r"$\mathrm{MAE}_{e_p,h}$  [m]",
        r"$\mathrm{MAE}_{e_v,h}$  [m/s]",
        r"$\mathrm{MAE}_{e_R,h}$  [rad]",
        r"$\mathrm{MAE}_{e_{\omega},h}$  [rad/s]",
    ]
    colors = plt.get_cmap("tab10").colors

    for idx, metric in enumerate(metric_names):
        ax = axs[idx]
        for exp_idx, experiment in enumerate(evaluated_experiments):
            metrics = experiment["metrics"][metric]
            horizons = np.array(sorted(metrics.keys()))
            values = np.array([metrics[h] for h in horizons])
            ax.plot(
                horizons,
                values,
                linewidth=2,
                color=colors[exp_idx % len(colors)],
                label=experiment["model_label"],
            )
        ax.set_ylabel(ylabels[idx], fontsize=12)
        ax.set_xlabel("$h$  [-]", fontsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=max(1, min(len(labels), 5)),
        bbox_to_anchor=(0.5, 1.12),
        fontsize=11,
    )
    plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.35)
    plt.show()


def show_prediction_plots(evaluated_experiments, horizon):
    reference_df = evaluated_experiments[0]["df_pred"]
    max_available = max(len(reference_df) - horizon, 1)
    n_start = min(2000, max_available - 1)
    n_end = min(n_start + 500, max_available)
    if n_end <= n_start:
        n_start = 0
        n_end = max_available

    state_names = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
    ]
    state_labels = [
        [r"$x$ [m]", r"$y$ [m]", r"$z$ [m]"],
        [r"$\varphi$ [rad]", r"$\theta$ [rad]", r"$\psi$ [rad]"],
        [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"],
        [r"$\omega_x$ [rad/s]", r"$\omega_y$ [rad/s]", r"$\omega_z$ [rad/s]"],
    ]

    fig, axs = plt.subplots(4, 3, figsize=(12, 5), sharex=True, dpi=200)
    t = reference_df["t"].values
    colors = plt.get_cmap("tab10").colors

    for r in range(4):
        for c in range(3):
            idx = r * 3 + c
            state = state_names[idx]
            pred_col = f"{state}_pred_h{horizon}"
            ax = axs[r, c]

            true = (
                reference_df[state][n_start + horizon : n_end + horizon]
                .rolling(20, min_periods=1, center=True)
                .mean()
            )

            for exp_idx, experiment in enumerate(evaluated_experiments):
                pred = (
                    experiment["df_pred"][pred_col][n_start:n_end]
                    .rolling(20, min_periods=1, center=True)
                    .mean()
                )
                ax.plot(
                    t[n_start + horizon : n_end + horizon],
                    pred,
                    linewidth=1.2,
                    color=colors[exp_idx % len(colors)],
                    label=experiment["model_label"],
                )

            ax.plot(
                t[n_start + horizon : n_end + horizon],
                true,
                "k--",
                linewidth=1.5,
                label="GT",
            )
            ax.set_ylabel(state_labels[r][c], fontsize=12)
            ax.grid(True, alpha=0.3)

            if r == 3 and c == 1:
                ax.set_xlabel("Time [s]", fontsize=12)

    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=max(1, min(len(labels), 6)),
        bbox_to_anchor=(0.5, 1.0),
        fontsize=11,
    )
    plt.subplots_adjust(top=0.84, bottom=0.12, hspace=0.28, wspace=0.35)
    plt.show()


def maybe_show_plots(evaluated_experiments, max_horizon):
    setup_matplotlib()
    show_metric_plots(evaluated_experiments)
    show_prediction_plots(evaluated_experiments, horizon=min(50, max_horizon))


def main(argv=None):
    args = parse_args(argv)
    experiments = build_experiments(args)
    evaluated_experiments = evaluate_experiments(experiments, args.max_horizon)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    new_rows_df = build_summary_df(evaluated_experiments, created_at=created_at)
    summary_path = Path(args.summary_path).expanduser().resolve()
    combined_df = append_summary(summary_path, new_rows_df)

    print(f"\n💾 Appended {len(new_rows_df)} row(s) to {summary_path}")
    print("\n=== Newly Appended Rows ===")
    print(new_rows_df.to_string(index=False))
    print(f"\n📚 Total rows in summary: {len(combined_df)}")

    print_latex_summary(evaluated_experiments, args.max_horizon)

    if args.show_plots:
        maybe_show_plots(evaluated_experiments, args.max_horizon)

    return combined_df


if __name__ == "__main__":
    main()
