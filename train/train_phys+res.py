import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset
from models.models import PhysQuadModel, PhysResQuadModel, ResidualQuadModel
from train.losses import WeightedMSELoss


def load_split(trajs, runs, base_dir, split, horizon):
    datasets = []
    for traj in trajs:
        for run in runs:
            file_name = f"{traj}_20251017_run{run}.csv"
            file_path = base_dir / file_name
            try:
                df = pd.read_csv(file_path)
                datasets.append(QuadDataset(df, horizon=horizon))
            except Exception as exc:
                print(f"⚠️ Skipped {file_path}: {exc}")

    if not datasets:
        raise RuntimeError(f"❌ No datasets could be loaded for {split} from {base_dir}")

    print(f"Loaded {len(datasets)} datasets for {split}")
    return datasets


parser = argparse.ArgumentParser(description="Train the physics+residual quadrotor model")
parser.add_argument("--train_trajs", type=str, default='["random", "square", "chirp"]')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--horizon", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--name-suffix", type=str, default="", help="Optional suffix appended to the auto-generated model name")
args = parser.parse_args()

train_trajs = json.loads(args.train_trajs)
valid_trajs = train_trajs
train_runs = [1, 2, 3]
valid_runs = [4]
device_str = args.device
epochs = args.epochs
horizon = args.horizon
batch_size = args.batch_size
dt = 0.01

model_name = f"phys+res_{'_'.join(train_trajs)}"
if args.name_suffix:
    model_name = f"{model_name}_{args.name_suffix}"
print(f"🧠 Model name composed automatically: {model_name}")

lr_start = 1e-5
lr_end = 1e-8

if device_str.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

model_dir = PROJECT_ROOT / "out" / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / f"{model_name}.pt"
print(f"✅ Model will be saved to: {model_path}")

scaler_dir = PROJECT_ROOT / "scalers" / model_name
scaler_dir.mkdir(parents=True, exist_ok=True)

data_dir = PROJECT_ROOT / "data" / "train"
train_ds = load_split(train_trajs, train_runs, data_dir, "train", horizon)
valid_ds = load_split(valid_trajs, valid_runs, data_dir, "valid", horizon)

train_dataset = combine_concat_dataset(
    ConcatDataset(train_ds), scale=True, fold="train", scaler_dir=scaler_dir
)
valid_dataset = combine_concat_dataset(
    ConcatDataset(valid_ds), scale=True, fold="valid", scaler_dir=scaler_dir
)

traj_info = {"train_trajs": train_trajs, "valid_trajs": valid_trajs}
traj_info_path = scaler_dir / "trajectories.json"
with open(traj_info_path, "w") as handle:
    json.dump(traj_info, handle, indent=4)
print(f"📝 Saved trajectory info to {traj_info_path}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
    "thrust_to_weight": 2.0,
    "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
}

phys_model = PhysQuadModel(phys_params, dt).to(device)
res_model = ResidualQuadModel(hidden_dim=64, num_layers=5, dt=dt).to(device)
model = PhysResQuadModel(
    phys=phys_model,
    residual=res_model,
    x_scaler=train_dataset.x_scaler,
    u_scaler=train_dataset.u_scaler,
).to(device)

print("🧩 Initialized PhysResQuadModel")

if args.pretrained and model_path.exists():
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"✅ Loaded pretrained model from {model_path}")
else:
    print("🔧 Training from scratch.")

optimizer = optim.Adam(model.parameters(), lr=lr_start)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)
criterion = WeightedMSELoss(lambda_=0.1)

best_val_loss = float("inf")
best_epoch = None

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x0, u_seq, x_seq in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
        x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
        optimizer.zero_grad()
        pred_seq = model(x0, u_seq)
        loss = criterion(pred_seq, x_seq)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x0, u_seq, x_seq in valid_loader:
            x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
            pred_seq = model(x0, u_seq)
            valid_loss += criterion(pred_seq, x_seq).item()

    avg_valid_loss = valid_loss / len(valid_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}, LR={current_lr:.2e}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        best_epoch = epoch + 1
        checkpoint = {
            "model_state": model.state_dict(),
            "config": {
                "state_dim": 12,
                "input_dim": 4,
                "hidden_dim": model.neural.hidden_dim,
                "num_layers": model.neural.num_layers,
                "dt": model.dt,
            },
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "train_trajs": train_trajs,
            "valid_trajs": valid_trajs,
        }
        torch.save(checkpoint, model_path)
        print(f"💾 Saved best model at epoch {best_epoch} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

print(f"✅ Training complete. Best model saved as {model_path} (epoch {best_epoch})")
