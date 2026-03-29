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


parser = argparse.ArgumentParser(description="Compare benchmark prediction results across models")
parser.add_argument("--train-trajs", type=str, default='["random", "square", "chirp"]')
parser.add_argument("--test-trajs", type=str, default='["melon"]')
parser.add_argument("--predictions-dir", type=str, default=str(PROJECT_ROOT / "out" / "predictions"))
parser.add_argument("--max-horizon", type=int, default=50)
parser.add_argument("--show-plots", action="store_true")
parser.add_argument("--summary-path", type=str, default=str(PROJECT_ROOT / "out" / "benchmark_summary.csv"))
args = parser.parse_args()

setup_matplotlib()

train_trajs = json.loads(args.train_trajs)
test_trajs = json.loads(args.test_trajs)
predictions_dir = Path(args.predictions_dir)
summary_path = Path(args.summary_path)
summary_path.parent.mkdir(parents=True, exist_ok=True)

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

    summary_rows.append({
        "model": model_name,
        "pos_h1": mm["pos"][1],
        "pos_h10": mm["pos"][10],
        "pos_h50": mm["pos"][target_horizons[-1]],
        "vel_h1": mm["vel"][1],
        "vel_h10": mm["vel"][10],
        "vel_h50": mm["vel"][target_horizons[-1]],
        "rot_h1": mm["rot"][1],
        "rot_h10": mm["rot"][10],
        "rot_h50": mm["rot"][target_horizons[-1]],
        "omega_h1": mm["omega"][1],
        "omega_h10": mm["omega"][10],
        "omega_h50": mm["omega"][target_horizons[-1]],
        "sim_pos": sim_p,
        "sim_vel": sim_v,
        "sim_rot": sim_r,
        "sim_omega": sim_w,
    })

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
