import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.latex_utils import print_latex_table_results
from utils.metrics_utils import compute_errors, compute_simerr
from utils.plot_utils import plot_metrics, plot_multistate_predictions, setup_matplotlib
from utils.quat_utils import quat_to_euler_np, so3_log_to_quat_np


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
        q = so3_log_to_quat_np(r)

        new_cols[f"qx{suffix}"] = q[:, 0]
        new_cols[f"qy{suffix}"] = q[:, 1]
        new_cols[f"qz{suffix}"] = q[:, 2]
        new_cols[f"qw{suffix}"] = q[:, 3]

        e = quat_to_euler_np(q)
        new_cols[f"roll{suffix}"] = e[:, 0]
        new_cols[f"pitch{suffix}"] = e[:, 1]
        new_cols[f"yaw{suffix}"] = e[:, 2]
        new_cols[f"roll{suffix}_deg"] = np.degrees(e[:, 0])
        new_cols[f"pitch{suffix}_deg"] = np.degrees(e[:, 1])
        new_cols[f"yaw{suffix}_deg"] = np.degrees(e[:, 2])

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def require_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing prediction file: {path}")
    return pd.read_csv(path)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary_row(model_name, metrics):
    sim_p, sim_v, sim_r, sim_w = compute_simerr(metrics)
    return {
        "model": model_name,
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


def build_full12_summary_row(experiment, metrics):
    row = build_summary_row(experiment["model_label"], metrics)
    row["model_label"] = experiment["model_label"]
    row["model_family"] = experiment["model_family"]
    row["loss_name"] = experiment["loss_name"]
    return row


def summarize_manifest(manifest_path, summary_path, max_horizon, expected_count=None):
    manifest = load_manifest(manifest_path)
    experiments = manifest.get("experiments", [])
    completed_experiments = [
        experiment
        for experiment in experiments
        if experiment.get("status") == "completed" and experiment.get("prediction_path")
    ]

    if not completed_experiments:
        raise RuntimeError("❌ No completed experiments with prediction files were found in the manifest")

    if expected_count is not None and len(completed_experiments) != expected_count:
        raise RuntimeError(
            "❌ Completed experiment count does not match expectation: "
            f"expected {expected_count}, found {len(completed_experiments)}"
        )

    summary_rows = []
    for experiment in completed_experiments:
        df_pred = add_rotation_columns(require_csv(Path(experiment["prediction_path"])))
        metrics = compute_errors(df_pred, max_horizon)
        summary_rows.append(build_full12_summary_row(experiment, metrics))

    summary_df = pd.DataFrame(summary_rows)
    column_order = [
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
    summary_df = summary_df[column_order]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Saved summary to {summary_path}")
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


parser = argparse.ArgumentParser(description="Compare benchmark prediction results across models")
parser.add_argument("--train-trajs", type=str, default='["random", "square", "chirp"]')
parser.add_argument("--test-trajs", type=str, default='["melon"]')
parser.add_argument("--predictions-dir", type=str, default=str(PROJECT_ROOT / "out" / "predictions"))
parser.add_argument("--max-horizon", type=int, default=50)
parser.add_argument("--show-plots", action="store_true")
parser.add_argument("--summary-path", type=str, default=str(PROJECT_ROOT / "out" / "benchmark_summary.csv"))
parser.add_argument("--manifest-path", type=str, default=None, help="Optional batch manifest for full-12 summary mode")
parser.add_argument("--expected-count", type=int, default=None, help="Optional expected number of completed experiments in manifest mode")
args = parser.parse_args()

setup_matplotlib()

train_trajs = json.loads(args.train_trajs)
test_trajs = json.loads(args.test_trajs)
predictions_dir = Path(args.predictions_dir)
summary_path = Path(args.summary_path)
summary_path.parent.mkdir(parents=True, exist_ok=True)

if args.max_horizon < 50:
    raise ValueError("--max-horizon must be at least 50 for the fixed *_h50 summary columns")

if args.manifest_path is not None:
    summarize_manifest(
        manifest_path=Path(args.manifest_path),
        summary_path=summary_path,
        max_horizon=args.max_horizon,
        expected_count=args.expected_count,
    )
    sys.exit(0)

train_name = "_".join(train_trajs)
test_name = "_".join(test_trajs)

file_lstm = predictions_dir / f"lstm_{train_name}_model_multistep" / f"{test_name}_multistep.csv"
file_base = predictions_dir / "baseline_model_multistep" / f"{test_name}_multistep.csv"
file_residual = predictions_dir / f"residual_{train_name}_model_multistep" / f"{test_name}_multistep.csv"
file_phys = predictions_dir / "physics_model_multistep" / f"{test_name}_multistep.csv"
file_physres = predictions_dir / f"phys+res_{train_name}_model_multistep" / f"{test_name}_multistep.csv"

print("LSTM file path:", file_lstm)
print("Baseline file path:", file_base)

df_lstm = require_csv(file_lstm)
df_base = require_csv(file_base)
df_residual = require_csv(file_residual)
df_phys = require_csv(file_phys)
df_physres = require_csv(file_physres)

print("✅ Loaded datasets:")
print(f"  LSTM model: {df_lstm.shape}")
print(f"  Baseline model: {df_base.shape}")
print(f"  Residual model: {df_residual.shape}")
print(f"  Physics model: {df_phys.shape}")
print(f"  Phys+Res model: {df_physres.shape}")

df_base, df_lstm, df_residual, df_phys, df_physres = [
    add_rotation_columns(df)
    for df in [df_base, df_lstm, df_residual, df_phys, df_physres]
]

metrics_base = compute_errors(df_base, args.max_horizon)
metrics_lstm = compute_errors(df_lstm, args.max_horizon)
metrics_residual = compute_errors(df_residual, args.max_horizon)
metrics_phys = compute_errors(df_phys, args.max_horizon)
metrics_physres = compute_errors(df_physres, args.max_horizon)

model_metrics = {
    "Naïve": metrics_base,
    "Physics": metrics_phys,
    "Residual": metrics_residual,
    "Phys+Res": metrics_physres,
    "LSTM": metrics_lstm,
}

if args.show_plots:
    plot_metrics(model_metrics, save_fig=False)

    dfs = {
        "Naive": df_base,
        "Physics": df_phys,
        "Residual": df_residual,
        "Phys+Res": df_physres,
        "LSTM": df_lstm,
    }
    n_start = 2000
    n_end = n_start + 500
    plot_multistate_predictions(dfs, h=min(50, args.max_horizon), N_start=n_start, N_end=n_end)

summary_rows = []
latex_rows = []
target_horizons = [1, 10, min(50, args.max_horizon)]
model_order = ["Naïve", "Physics", "Residual", "Phys+Res", "LSTM"]

for model_name in model_order:
    mm = model_metrics[model_name]
    sim_p, sim_v, sim_r, sim_w = compute_simerr(mm)

    summary_row = build_summary_row(model_name, mm)
    summary_row["pos_h50"] = mm["pos"][target_horizons[-1]]
    summary_row["vel_h50"] = mm["vel"][target_horizons[-1]]
    summary_row["rot_h50"] = mm["rot"][target_horizons[-1]]
    summary_row["omega_h50"] = mm["omega"][target_horizons[-1]]
    summary_rows.append(summary_row)

    latex_rows.append([
        model_name,
        mm["pos"][1], mm["pos"][10], mm["pos"][target_horizons[-1]], sim_p,
        mm["vel"][1], mm["vel"][10], mm["vel"][target_horizons[-1]], sim_v,
        mm["rot"][1], mm["rot"][10], mm["rot"][target_horizons[-1]], sim_r,
        mm["omega"][1], mm["omega"][10], mm["omega"][target_horizons[-1]], sim_w,
    ])

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_path, index=False)
print(f"\n💾 Saved summary to {summary_path}")
print("\n=== Summary ===")
print(summary_df.to_string(index=False))

print_latex_table_results(latex_rows, target_horizons)
