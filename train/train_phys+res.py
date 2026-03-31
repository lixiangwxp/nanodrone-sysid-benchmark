import argparse
import json
import os
import sys
import time
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
from utils.seed_utils import build_torch_generator, seed_worker, set_global_seed
from utils.wandb_utils import (
    add_wandb_args,
    compute_gradient_norm,
    compute_parameter_norm,
    finish_run,
    init_wandb_run,
    log_metrics,
    maybe_watch_model,
)


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
parser.add_argument("--seed", type=int, default=42, help="Random seed used for initialization and DataLoader shuffling")
add_wandb_args(parser)
args = parser.parse_args()

train_trajs = json.loads(args.train_trajs)
valid_trajs = train_trajs
train_runs = [1, 2, 3]
valid_runs = [4]
device_str = args.device
epochs = args.epochs
horizon = args.horizon
batch_size = args.batch_size
seed = args.seed
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

set_global_seed(seed)
print(f"🌱 Global seed set to: {seed}")

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

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=build_torch_generator(seed),
    worker_init_fn=seed_worker,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    generator=build_torch_generator(seed),
    worker_init_fn=seed_worker,
)

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
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_trainable_params:,}")

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

wandb_run = init_wandb_run(
    args=args,
    project_root=PROJECT_ROOT,
    model_name=model_name,
    config={
        **vars(args),
        "model_type": "phys+res",
        "train_trajs": train_trajs,
        "valid_trajs": valid_trajs,
        "train_runs": train_runs,
        "valid_runs": valid_runs,
        "lr_start": lr_start,
        "lr_end": lr_end,
        "dt": dt,
        "device": str(device),
        "model_path": model_path,
        "scaler_dir": scaler_dir,
        "num_trainable_params": num_trainable_params,
        "train_dataset_size": len(train_dataset),
        "valid_dataset_size": len(valid_dataset),
        "phys_params": phys_params,
    },
    tags=["physres", *train_trajs],
    group=f"physres__{'_'.join(train_trajs)}",
)
maybe_watch_model(wandb_run, model, args)

best_val_loss = float("inf")
best_epoch = None

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    train_loss = 0.0
    train_grad_norm_total = 0.0
    train_grad_norm_max = 0.0
    for x0, u_seq, x_seq in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
        x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
        optimizer.zero_grad()
        pred_seq = model(x0, u_seq)
        loss = criterion(pred_seq, x_seq)
        loss.backward()
        grad_norm = compute_gradient_norm(model)
        train_grad_norm_total += grad_norm
        train_grad_norm_max = max(train_grad_norm_max, grad_norm)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_grad_norm = train_grad_norm_total / len(train_loader)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x0, u_seq, x_seq in valid_loader:
            x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
            pred_seq = model(x0, u_seq)
            valid_loss += criterion(pred_seq, x_seq).item()

    avg_valid_loss = valid_loss / len(valid_loader)
    current_lr = scheduler.get_last_lr()[0]
    is_best = False
    print(f"Epoch {epoch + 1}, LR={current_lr:.2e}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        best_epoch = epoch + 1
        is_best = True
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
            "seed": seed,
        }
        torch.save(checkpoint, model_path)
        print(f"💾 Saved best model at epoch {best_epoch} with valid loss {avg_valid_loss:.6f}")

    log_metrics(
        wandb_run,
        {
            "epoch": epoch + 1,
            "train/loss_total": avg_train_loss,
            "valid/loss_total": avg_valid_loss,
            "train/grad_norm_mean": avg_train_grad_norm,
            "train/grad_norm_max": train_grad_norm_max,
            "model/param_norm": compute_parameter_norm(model),
            "optim/lr": current_lr,
            "best/val_loss": best_val_loss,
            "best/epoch": best_epoch or 0,
            "checkpoint/is_best": int(is_best),
            "timing/epoch_sec": time.time() - epoch_start,
        },
    )

    scheduler.step()

print(f"✅ Training complete. Best model saved as {model_path} (epoch {best_epoch})")
finish_run(
    wandb_run,
    summary={
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_path": model_path,
    },
)
