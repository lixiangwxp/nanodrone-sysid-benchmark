import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset


parser = argparse.ArgumentParser(description="Evaluate the naive constant-state baseline")
parser.add_argument("--test-trajs", type=str, default='["melon"]')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--horizon", type=int, default=50)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
horizon = args.horizon
dt = 0.01
model_name = "baseline"

test_trajs = json.loads(args.test_trajs)
data_dir = PROJECT_ROOT / "data" / "test"

print(f"🧪 Test trajectories (auto-selected): {test_trajs}")

# ---------------------------------------------------------------------
# === Load test datasets ===
# ---------------------------------------------------------------------
test_ds = []
for traj in test_trajs:
    for run in [1, 2, 3]:
        file_name = f"{traj}_20251017_run{run}.csv"
        file_path = data_dir / file_name
        try:
            df = pd.read_csv(file_path)
            ds = QuadDataset(df, horizon=horizon)
            test_ds.append(ds)
        except Exception as exc:
            print(f"⚠️ Skipped {file_name}: {exc}")

if not test_ds:
    raise RuntimeError(f"❌ No test datasets could be loaded from {data_dir}")

test_dataset = combine_concat_dataset(
    ConcatDataset(test_ds), scale=False
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

# ---------------------------------------------------------------------
# === Run predictions ===
# ---------------------------------------------------------------------
preds, trues = [], []
with torch.no_grad():
    for x0, u_seq, x_seq_true in test_loader:
        x0, u_seq = x0.to(device), u_seq.to(device)
        x_pred = x0.repeat(1, horizon, 1).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

# =====================================================
# --- Convert to DataFrame ---
# =====================================================
N = preds.shape[0]
data = {}

if hasattr(test_dataset, "df") and "t" in test_dataset.df.columns:
    t_vec = test_dataset.df["t"].values[:N]
else:
    t_vec = np.arange(N) * dt
data["t"] = t_vec

for i, name in enumerate(state_names):
    data[name] = trues[:, 0, i]

for h in range(1, horizon + 1):
    for i, name in enumerate(state_names):
        data[f"{name}_pred_h{h}"] = preds[:, h - 1, i]

df_pred = pd.DataFrame(data)
print(f"✅ Prediction DataFrame shape: {df_pred.shape}")

# =====================================================
# --- Save results ---
# =====================================================
out_dir = PROJECT_ROOT / "out" / "predictions" / "baseline_model_multistep"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{'_'.join(test_trajs)}_multistep.csv"
df_pred.to_csv(out_path, index=False)
print(f"💾 Saved to {out_path}")
