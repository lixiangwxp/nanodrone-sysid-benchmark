import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.onnx import TrainingMode
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset
from models.models import QuadLSTM


def select_model_index(num_models, requested_index):
    if requested_index is not None:
        if 1 <= requested_index <= num_models:
            return requested_index
        raise ValueError(f"model-index must be between 1 and {num_models}")

    if not sys.stdin.isatty():
        print("\nℹ️ Non-interactive stdin detected, defaulting to model [1].")
        return 1

    while True:
        try:
            choice = int(input(f"\nSelect model [1–{num_models}]: ").strip())
            if 1 <= choice <= num_models:
                return choice
            print(f"⚠️ Please enter a number between 1 and {num_models}.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a valid number.")


def infer_lstm_config(state_dict, dt):
    hidden_dim, state_dim_x = state_dict["h0_net.0.weight"].shape
    input_dim_u = state_dict["lstm.weight_ih_l0"].shape[1]
    num_layers = len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])
    return {
        "input_dim_u": input_dim_u,
        "state_dim_x": state_dim_x,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dt": dt,
    }


parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model on the test trajectories")
parser.add_argument("--model-index", type=int, default=None, help="1-based index in the displayed model list")
parser.add_argument("--test-trajs", type=str, default='["melon"]')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--horizon", type=int, default=50)
parser.add_argument("--show-plot", action="store_true")
parser.add_argument("--export-onnx", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
horizon = args.horizon
dt = 0.01
test_trajs = json.loads(args.test_trajs)
data_dir = PROJECT_ROOT / "data" / "test"

# ---------------------------------------------------------------------
# === Locate trained model automatically ===
# ---------------------------------------------------------------------
model_root = PROJECT_ROOT / "out" / "models"
model_files = sorted(
    [f.name for f in model_root.iterdir() if f.is_file() and f.name.startswith("lstm") and f.suffix == ".pt"],
    key=lambda x: os.path.getmtime(model_root / x),
    reverse=True,
)

if not model_files:
    raise RuntimeError(f"❌ No trained model found in {model_root}")

print("\n📂 Available trained models:")
for idx, name in enumerate(model_files, start=1):
    mtime = os.path.getmtime(model_root / name)
    print(f"  [{idx}] {name}  (modified: {pd.to_datetime(mtime, unit='s'):%Y-%m-%d %H:%M})")

choice = select_model_index(len(model_files), args.model_index)
model_file = model_files[choice - 1]
model_name = model_file.replace(".pt", "")
model_path = model_root / model_file

print(f"\n✅ Selected model: {model_name}")

# ---------------------------------------------------------------------
# === Load training trajectory info ===
# ---------------------------------------------------------------------
scaler_dir = PROJECT_ROOT / "scalers" / model_name
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
    ConcatDataset(test_ds), scale=True, fold="test", scaler_dir=scaler_dir
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

# ---------------------------------------------------------------------
# === Load trained model ===
# ---------------------------------------------------------------------
ckpt = torch.load(model_path, map_location=device)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
cfg = ckpt.get("config", infer_lstm_config(state, dt)) if isinstance(ckpt, dict) else infer_lstm_config(state, dt)
model = QuadLSTM(**cfg).to(device)
model.load_state_dict(state)
model.eval()
print(f"✅ Model loaded from {model_path}")

# ---------------------------------------------------------------------
# === Run predictions ===
# ---------------------------------------------------------------------
preds, trues = [], []
with torch.no_grad():
    for x0, u_seq, x_seq_true in test_loader:
        x0, u_seq = x0.to(device), u_seq.to(device)
        x_pred = model(x0, u_seq).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

# ---------------------------------------------------------------------
# === Denormalize ===
# ---------------------------------------------------------------------
x_scaler = joblib.load(scaler_dir / "x_scaler.pkl")
preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

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
out_dir = PROJECT_ROOT / "out" / "predictions" / f"{model_name}_model_multistep"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{'_'.join(test_trajs)}_multistep.csv"
df_pred.to_csv(out_path, index=False)
print(f"💾 Saved to {out_path}")

# =====================================================
# --- Quick sanity check plot ---
# =====================================================
plt.figure(figsize=(8, 4))
plt.plot(df_pred["t"], df_pred["x"], label="x true")
plt.plot(df_pred["t"], df_pred["x_pred_h1"], "--", label="x pred (h=1)")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
if args.show_plot:
    plt.show()
else:
    plt.close()

# =====================================================================
# === OPTIONAL: EXPORT LSTM ONE-STEP MODEL FOR PROFILING ===============
# =====================================================================
onnx_export_dir = PROJECT_ROOT / "out" / "export" / f"{model_name}_model_multistep"
onnx_export_dir.mkdir(parents=True, exist_ok=True)

if args.export_onnx:
    print("📦 Exporting 1-step LSTM model for profiling...")

    class LSTMOneStep(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x0, u0):
            u_seq = u0.unsqueeze(1)
            x_seq = self.base(x0, u_seq)
            return x_seq[:, 0]

    one_step_model = LSTMOneStep(model).eval().to(device)

    sample_x0 = torch.tensor(preds[0, 0], dtype=torch.float32)
    sample_u0 = torch.zeros(4, dtype=torch.float32)

    torch.save(
        {"x0": sample_x0, "u0": sample_u0},
        onnx_export_dir / "sample_io.pt"
    )
    print("💾 Saved sample I/O → sample_io.pt")

    dummy_x0 = sample_x0.unsqueeze(0).to(device)
    dummy_u0 = sample_u0.unsqueeze(0).to(device)
    onnx_path = onnx_export_dir / f"{model_name}_1step.onnx"

    try:
        torch.onnx.export(
            one_step_model,
            (dummy_x0, dummy_u0),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["x0", "u0"],
            output_names=["x1"],
            dynamic_axes={
                "x0": {0: "batch_size"},
                "u0": {0: "batch_size"},
                "x1": {0: "batch_size"},
            },
            training=TrainingMode.TRAINING,
        )
        print(f"🟢 ONNX 1-step LSTM exported → {onnx_path}")
    except Exception as exc:
        print(f"❌ ONNX export failed: {exc}")
