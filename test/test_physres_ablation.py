import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import QuadDataset, combine_concat_dataset
from models.models import PhysQuadModel, PhysResQuadModel, ResidualQuadModel
from models.models_lag import (
    LagPhysResGRUControlModel,
    LagPhysResGRUForceModel,
    LagPhysResGRUModel,
    LagPhysResGRUTorqueModel,
    LagPhysResQuadModel,
)


def parse_json_list(raw_value, arg_name):
    parsed = json.loads(raw_value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must decode to a list or string")
    return parsed


def uses_lag(variant):
    return variant in {"lag", "lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force", "lag_geo", "full"}


def uses_hist_init(variant):
    return variant == "lag_gru_histinit_honly"


def find_latest_model(model_root):
    model_files = sorted(
        [
            path
            for path in model_root.iterdir()
            if path.is_file() and path.name.startswith("physres_ablation__") and path.suffix == ".pt"
        ],
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    if not model_files:
        raise RuntimeError(f"❌ No physres_ablation checkpoint found in {model_root}")
    return model_files[0]


def load_test_datasets(test_trajs, data_root, horizon, history_len=0, start_offset=0):
    datasets = []
    for traj in test_trajs:
        for run in [1, 2, 3]:
            file_name = f"{traj}_20251017_run{run}.csv"
            file_path = data_root / file_name
            try:
                df = pd.read_csv(file_path)
                datasets.append(
                    QuadDataset(
                        df,
                        horizon=horizon,
                        history_len=history_len,
                        start_offset=start_offset,
                    )
                )
            except Exception as exc:
                print(f"⚠️ Skipped {file_path}: {exc}")

    if not datasets:
        raise RuntimeError(f"❌ No test datasets could be loaded from {data_root}")

    return datasets


def rebuild_model(checkpoint, x_scaler, u_scaler, device):
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing 'config'")
    if "phys_params" not in checkpoint:
        raise KeyError("Checkpoint is missing 'phys_params'")

    config = checkpoint["config"]
    phys_params = checkpoint["phys_params"]
    variant = config["variant"]
    hidden_dim = config.get("hidden_dim", 64)
    gru_hidden_dim = config.get("gru_hidden_dim", 64)
    num_layers = config.get("num_layers", 5)
    default_residual_input_dim = 4
    if variant in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"}:
        default_residual_input_dim = gru_hidden_dim + 12
    elif uses_lag(variant):
        default_residual_input_dim = 12
    residual_input_dim = config.get("residual_input_dim", default_residual_input_dim)
    dt = config.get("dt", 0.01)

    phys_model = PhysQuadModel(phys_params, dt).to(device)
    residual_model = ResidualQuadModel(
        input_dim=residual_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dt=dt,
    ).to(device)

    if variant == "lag_gru":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
        ).to(device)
    elif variant == "lag_gru_uinit":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
            alpha_dim=1,
            use_u_init=True,
            u_init_scale=0.05,
        ).to(device)
    elif variant == "lag_gru_histinit_honly":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
            alpha_dim=1,
            use_u_init=False,
            u_init_scale=0.05,
            use_hist_init=True,
            hist_init_scale=config.get("hist_init_scale", 0.1),
        ).to(device)
    elif variant == "lag_gru_alpha4":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
            alpha_dim=4,
        ).to(device)
    elif variant == "lag_gru_ctrl":
        model = LagPhysResGRUControlModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
            control_ctx_dim=config.get("control_ctx_dim", 32),
        ).to(device)
    elif variant == "lag_gru_torque":
        model = LagPhysResGRUTorqueModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
            torque_scale_factor=config.get("torque_scale_factor", 0.2),
        ).to(device)
    elif variant == "lag_gru_force":
        model = LagPhysResGRUForceModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            hidden_dim=gru_hidden_dim,
        ).to(device)
    elif uses_lag(variant):
        model = LagPhysResQuadModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=config.get("lag_mode", "per_motor"),
            alpha_init=config.get("alpha_init", 0.85),
            use_aux_head=config.get("use_acc_aux", False),
            aux_dim=config.get("aux_dim", 1),
        ).to(device)
    else:
        model = PhysResQuadModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
        ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate unified PhysRes ablation checkpoints")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--test_trajs", type=str, default='["melon"]')
    parser.add_argument("--out_root", type=str, default=str(PROJECT_ROOT / "out" / "predictions"))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data" / "test"))
    args = parser.parse_args()
    if args.history_len < 0:
        raise ValueError("--history-len must be >= 0")
    if args.start_offset < 0:
        raise ValueError("--start-offset must be >= 0")

    test_trajs = parse_json_list(args.test_trajs, "test_trajs")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dt = 0.01
    batch_size = 128

    model_root = PROJECT_ROOT / "out" / "models"
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        model_path = find_latest_model(model_root)
    model_path = model_path.resolve()
    model_name = model_path.stem

    checkpoint = torch.load(model_path, map_location=device)
    variant = checkpoint["config"]["variant"]
    history_len = checkpoint["config"].get("history_len", args.history_len)
    if uses_hist_init(variant) and history_len <= 0:
        raise ValueError(
            "lag_gru_histinit_honly requires history_len from checkpoint config "
            "or --history-len > 0"
        )
    effective_start_offset = max(args.start_offset, history_len) if uses_hist_init(variant) else args.start_offset

    scaler_dir_str = checkpoint.get("scaler_dir")
    if scaler_dir_str is not None and Path(scaler_dir_str).exists():
        scaler_dir = Path(scaler_dir_str)
    else:
        scaler_dir = PROJECT_ROOT / "scalers" / model_name

    data_root = Path(args.data_root)
    test_ds = load_test_datasets(
        test_trajs,
        data_root,
        args.horizon,
        history_len=history_len if uses_hist_init(variant) else 0,
        start_offset=effective_start_offset,
    )
    test_dataset = combine_concat_dataset(
        ConcatDataset(test_ds), scale=True, fold="test", scaler_dir=scaler_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = rebuild_model(
        checkpoint=checkpoint,
        x_scaler=test_dataset.x_scaler,
        u_scaler=test_dataset.u_scaler,
        device=device,
    )

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x0, u_seq, x_seq_true = batch[:3]
            x0 = x0.to(device)
            u_seq = u_seq.to(device)

            if uses_hist_init(variant):
                x_hist, u_hist = batch[3:5]
                x_hist = x_hist.to(device)
                u_hist = u_hist.to(device)
                x_pred = model(x0, u_seq, x_hist=x_hist, u_hist=u_hist).cpu()
            else:
                x_pred = model(x0, u_seq).cpu()
            preds.append(x_pred)
            trues.append(x_seq_true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    x_scaler = test_dataset.x_scaler
    preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)
    start_indices = (
        test_dataset.start_indices.cpu().numpy()
        if getattr(test_dataset, "start_indices", None) is not None
        else np.arange(preds.shape[0])
    )

    state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

    num_windows = preds.shape[0]
    if len(start_indices) != num_windows:
        raise ValueError("Prediction windows and start indices are misaligned")
    data = {"start_idx": start_indices, "t": start_indices * dt}
    for idx, name in enumerate(state_names):
        data[name] = trues[:, 0, idx]

    for h in range(1, args.horizon + 1):
        for idx, name in enumerate(state_names):
            data[f"{name}_pred_h{h}"] = preds[:, h - 1, idx]

    df_pred = pd.DataFrame(data)

    out_root = Path(args.out_root)
    out_dir = out_root / f"{model_name}_model_multistep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{'_'.join(test_trajs)}_multistep.csv"
    df_pred.to_csv(out_path, index=False)

    print(f"✅ Loaded model: {model_path}")
    print(f"🧩 Variant: {variant}")
    print(f"💾 Saved predictions to: {out_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(df_pred["t"], df_pred["x"], label="x true")
    plt.plot(df_pred["t"], df_pred["x_pred_h1"], "--", label="x pred (h=1)")
    plt.xlabel("Time [s]")
    plt.ylabel("x [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    main()
