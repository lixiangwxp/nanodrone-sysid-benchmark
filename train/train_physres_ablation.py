import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
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
from dataset.dataset_aux import QuadDatasetWithAux, combine_concat_dataset_with_aux
from models.models import PhysQuadModel, PhysResQuadModel, ResidualQuadModel
from models.models_lag import (
    LagPhysResGRUForceModel,
    LagPhysResGRUModel,
    LagPhysResQuadModel,
)
from train.losses import WeightedMSELoss
from train.losses_ext import CompositeAblationLoss
from utils.seed_utils import build_torch_generator, seed_worker, set_global_seed
from utils.wandb_utils import (
    add_wandb_args,
    collect_lag_metrics,
    compute_gradient_norm,
    compute_parameter_norm,
    finish_run,
    init_wandb_run,
    log_metrics,
    maybe_watch_model,
    prefix_metrics,
)


VARIANT_CHOICES = ["baseline", "geo", "lag", "lag_gru", "lag_gru_force", "lag_geo", "full"]
DEFAULT_AUX_COLS_RAW = '["az_body"]'
DEFAULT_FORCE_AUX_COLS = ["ax_body", "ay_body", "az_body"]


def parse_json_list(raw_value, arg_name):
    parsed = json.loads(raw_value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must decode to a list or string")
    return parsed


def uses_lag(variant):
    return variant in {"lag", "lag_gru", "lag_gru_force", "lag_geo", "full"}


def uses_geo_loss(variant):
    return variant in {"geo", "lag_geo", "full"}


def uses_aux_supervision(variant):
    return variant == "full"


def uses_force_supervision(variant):
    return variant == "lag_gru_force"


def uses_aux_dataset(variant):
    return uses_aux_supervision(variant) or uses_force_supervision(variant)


def uses_plain_temporal_loss(variant):
    return variant in {"baseline", "lag", "lag_gru", "lag_gru_force"}


def build_model_name(variant, train_trajs):
    return f"physres_ablation__{variant}__{'_'.join(train_trajs)}"


def build_phys_params():
    return {
        "g": 9.81,
        "m": 0.045,
        "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
        "thrust_to_weight": 2.0,
        "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
    }


def load_split(trajs, runs, data_dir, split, horizon, use_aux=False, aux_cols=None):
    datasets = []
    for traj in trajs:
        for run in runs:
            file_name = f"{traj}_20251017_run{run}.csv"
            file_path = data_dir / file_name
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                print(f"⚠️ Skipped {file_path}: {exc}")
                continue

            if use_aux:
                ds = QuadDatasetWithAux(
                    df,
                    horizon=horizon,
                    aux_cols=aux_cols,
                    use_acc_aux=True,
                )
            else:
                try:
                    ds = QuadDataset(df, horizon=horizon)
                except Exception as exc:
                    print(f"⚠️ Skipped {file_path}: {exc}")
                    continue

            datasets.append(ds)

    if not datasets:
        raise RuntimeError(f"❌ No datasets could be loaded for {split} from {data_dir}")

    print(f"Loaded {len(datasets)} datasets for {split}")
    return datasets


def build_model(
    variant,
    phys_params,
    dt,
    x_scaler,
    u_scaler,
    lag_mode,
    alpha_init,
    aux_dim,
    hidden_dim,
    gru_hidden_dim,
):
    num_layers = 5
    if variant in {"lag_gru", "lag_gru_force"}:
        residual_input_dim = gru_hidden_dim + 12
    elif uses_lag(variant):
        residual_input_dim = 12
    else:
        residual_input_dim = 4

    phys_model = PhysQuadModel(phys_params, dt)
    residual_model = ResidualQuadModel(
        input_dim=residual_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dt=dt,
    )

    if variant == "lag_gru":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
        )
    elif variant == "lag_gru_force":
        model = LagPhysResGRUForceModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
        )
    elif uses_lag(variant):
        model = LagPhysResQuadModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            use_aux_head=uses_aux_supervision(variant),
            aux_dim=aux_dim,
        )
    else:
        model = PhysResQuadModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
        )

    return model, hidden_dim, num_layers, residual_input_dim


def build_criterion(variant, x_scaler, beta_geo, beta_aux, w_rot, w_omega):
    if uses_plain_temporal_loss(variant):
        return WeightedMSELoss(lambda_=0.1)

    return CompositeAblationLoss(
        x_scaler=x_scaler,
        lambda_mse=0.1,
        use_geo=uses_geo_loss(variant),
        beta_geo=beta_geo,
        use_aux=uses_aux_supervision(variant),
        beta_aux=beta_aux,
        aux_loss_type="smooth_l1",
        w_rot=w_rot,
        w_omega=w_omega,
    )


def init_metric_totals():
    return {
        "loss_total": 0.0,
        "loss_mse": 0.0,
        "loss_geo": 0.0,
        "loss_aux": 0.0,
    }


def average_metrics(metric_totals, num_batches):
    return {key: value / num_batches for key, value in metric_totals.items()}


def compute_mixed_temporal_loss(pred_seq, x_seq):
    loss_exp = WeightedMSELoss(lambda_=0.03)(pred_seq, x_seq)
    loss_uniform = torch.mean((pred_seq - x_seq) ** 2)
    loss_tail = torch.mean((pred_seq[:, -10:] - x_seq[:, -10:]) ** 2)
    total_loss = 0.5 * loss_exp + 0.3 * loss_uniform + 0.2 * loss_tail

    zero = total_loss.detach().new_tensor(0.0)
    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_mse": total_loss.detach(),
        "loss_geo": zero,
        "loss_aux": zero,
        "loss_exp": loss_exp.detach(),
        "loss_uniform": loss_uniform.detach(),
        "loss_tail": loss_tail.detach(),
    }
    return total_loss, loss_dict


def maybe_compute_force_loss(model, batch, force_pred_seq, u_eff_seq_real, device, beta_force):
    if len(batch) < 4:
        zero = force_pred_seq.detach().new_tensor(0.0)
        return zero, zero

    aux_seq = batch[3].to(device)
    if aux_seq.shape[-1] < 3:
        zero = force_pred_seq.detach().new_tensor(0.0)
        return zero, zero

    f_meas_body = model.phys.m * aux_seq[..., :3]
    thrust = model.phys.Kt * (u_eff_seq_real ** 2).sum(dim=-1, keepdim=True)
    zeros = torch.zeros_like(thrust)
    f_phys_body = torch.cat([zeros, zeros, thrust], dim=-1)
    force_target_seq = f_meas_body - f_phys_body

    force_loss = torch.nn.functional.smooth_l1_loss(force_pred_seq, force_target_seq)
    weighted_force_loss = beta_force * force_loss
    return weighted_force_loss, force_loss.detach()


def compute_loss(model, criterion, batch, device, variant, loss_type):
    if variant == "lag_gru_force":
        x0, u_seq, x_seq = batch[:3]
        x0 = x0.to(device)
        u_seq = u_seq.to(device)
        x_seq = x_seq.to(device)

        pred_seq, force_pred_seq, u_eff_seq_real = model(x0, u_seq, return_force=True)
        mixed_loss, loss_dict = compute_mixed_temporal_loss(pred_seq, x_seq)
        weighted_force_loss, force_loss_value = maybe_compute_force_loss(
            model=model,
            batch=batch,
            force_pred_seq=force_pred_seq,
            u_eff_seq_real=u_eff_seq_real,
            device=device,
            beta_force=getattr(model, "beta_force", 0.0),
        )
        total_loss = mixed_loss + weighted_force_loss
        loss_dict["loss_total"] = total_loss.detach()
        loss_dict["loss_mse"] = mixed_loss.detach()
        loss_dict["loss_force"] = force_loss_value
        return total_loss, loss_dict

    use_aux = uses_aux_supervision(variant)

    if use_aux:
        x0, u_seq, x_seq, aux_seq = batch
        x0 = x0.to(device)
        u_seq = u_seq.to(device)
        x_seq = x_seq.to(device)
        aux_seq = aux_seq.to(device)
        pred_seq, aux_pred = model(x0, u_seq, return_aux=True)
        total_loss, loss_dict = criterion(pred_seq, x_seq, aux_pred=aux_pred, aux_true=aux_seq)
        return total_loss, loss_dict

    x0, u_seq, x_seq = batch
    x0 = x0.to(device)
    u_seq = u_seq.to(device)
    x_seq = x_seq.to(device)
    pred_seq = model(x0, u_seq)

    if uses_plain_temporal_loss(variant) and loss_type == "mixed":
        return compute_mixed_temporal_loss(pred_seq, x_seq)

    if isinstance(criterion, CompositeAblationLoss):
        total_loss, loss_dict = criterion(pred_seq, x_seq)
        return total_loss, loss_dict

    total_loss = criterion(pred_seq, x_seq)
    zero = total_loss.detach().new_tensor(0.0)
    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_mse": total_loss.detach(),
        "loss_geo": zero,
        "loss_aux": zero,
    }
    return total_loss, loss_dict


def build_autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=torch.float16)


def update_metric_totals(metric_totals, loss_dict):
    for key, value in loss_dict.items():
        metric_totals.setdefault(key, 0.0)
        metric_totals[key] += value.item()


def build_optimizer(model, variant, base_lr):
    weight_decay = 1e-5
    if variant not in {"lag_gru", "lag_gru_force"}:
        return optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    grouped_ids = set()

    def collect_params(module):
        params = []
        if module is None:
            return params
        for param in module.parameters():
            if not param.requires_grad or id(param) in grouped_ids:
                continue
            params.append(param)
            grouped_ids.add(id(param))
        return params

    fast_params = collect_params(getattr(model, "u_init_head", None))
    fast_params.extend(collect_params(getattr(model, "alpha_head", None)))

    medium_params = collect_params(getattr(model, "gru_cell", None))
    medium_params.extend(collect_params(getattr(model, "force_head", None)))
    medium_params.extend(collect_params(getattr(model, "residual", None)))
    medium_params.extend(collect_params(getattr(model, "h_init", None)))

    slow_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in grouped_ids
    ]

    param_groups = []
    if fast_params:
        param_groups.append({"params": fast_params, "lr": 3e-4, "weight_decay": weight_decay})
    if medium_params:
        param_groups.append({"params": medium_params, "lr": 1e-4, "weight_decay": weight_decay})
    if slow_params:
        param_groups.append({"params": slow_params, "lr": base_lr, "weight_decay": weight_decay})

    return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)


def build_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_metrics,
    val_metrics,
    best_val_loss,
    best_epoch,
    variant,
    hidden_dim,
    num_layers,
    residual_input_dim,
    lag_mode,
    alpha_init,
    aux_cols,
    aux_dim,
    beta_geo,
    beta_aux,
    beta_force,
    w_rot,
    w_omega,
    phys_params,
    dt,
    horizon,
    total_epochs,
    lr_start,
    lr_end,
    batch_size,
    gru_hidden_dim,
    loss_type,
    amp_enabled,
    scaler_dir,
    train_trajs,
    valid_trajs,
    seed,
):
    return {
        "model_state": model.state_dict(),
        "config": {
            "variant": variant,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "use_lag": uses_lag(variant),
            "use_geo_loss": uses_geo_loss(variant),
            "use_acc_aux": uses_aux_supervision(variant),
            "lag_mode": lag_mode,
            "alpha_init": alpha_init,
            "aux_cols": aux_cols,
            "aux_dim": aux_dim,
            "beta_geo": beta_geo,
            "beta_aux": beta_aux,
            "beta_force": beta_force,
            "w_rot": w_rot,
            "w_omega": w_omega,
            "residual_input_dim": residual_input_dim,
            "dt": dt,
            "horizon": horizon,
            "total_epochs": total_epochs,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "batch_size": batch_size,
            "gru_hidden_dim": gru_hidden_dim if variant in {"lag_gru", "lag_gru_force"} else None,
            "loss_type": loss_type,
            "amp": amp_enabled,
        },
        "phys_params": phys_params,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "completed_epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_loss": train_metrics["loss_total"],
        "val_loss": val_metrics["loss_total"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_trajs": train_trajs,
        "valid_trajs": valid_trajs,
        "seed": seed,
        "model_name": build_model_name(variant, train_trajs),
        "scaler_dir": str(scaler_dir),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def resolve_checkpoint_path(raw_path, model_dir, model_name):
    if raw_path is None:
        return None
    if raw_path == "latest":
        return model_dir / f"{model_name}__latest.pt"
    if raw_path == "best":
        return model_dir / f"{model_name}.pt"
    return Path(raw_path).expanduser()


def validate_resume_checkpoint(checkpoint, args, train_trajs, valid_trajs, aux_cols):
    if "model_state" not in checkpoint:
        raise KeyError("Resume checkpoint is missing 'model_state'")
    if "config" not in checkpoint:
        raise KeyError("Resume checkpoint is missing 'config'")

    config = checkpoint["config"]
    mismatches = []

    expected_pairs = [
        ("variant", args.variant, config.get("variant")),
        ("horizon", args.horizon, config.get("horizon")),
        ("epochs", args.epochs, config.get("total_epochs")),
        ("train_trajs", train_trajs, checkpoint.get("train_trajs")),
        ("valid_trajs", valid_trajs, checkpoint.get("valid_trajs")),
    ]

    if config.get("loss_type") is not None:
        expected_pairs.append(("loss_type", args.loss_type, config.get("loss_type")))
    if config.get("lr_start") is not None:
        expected_pairs.append(("lr_start", args.lr_start, config.get("lr_start")))
    if config.get("lr_end") is not None:
        expected_pairs.append(("lr_end", args.lr_end, config.get("lr_end")))
    if config.get("hidden_dim") is not None:
        expected_pairs.append(("hidden_dim", args.hidden_dim, config.get("hidden_dim")))
    if args.variant in {"lag_gru", "lag_gru_force"} and config.get("gru_hidden_dim") is not None:
        expected_pairs.append(("gru_hidden_dim", args.gru_hidden_dim, config.get("gru_hidden_dim")))

    if config.get("beta_geo") is not None:
        expected_pairs.append(("beta_geo", args.beta_geo, config.get("beta_geo")))
    if config.get("beta_force") is not None:
        expected_pairs.append(("beta_force", args.beta_force, config.get("beta_force")))
    if config.get("w_rot") is not None:
        expected_pairs.append(("w_rot", args.w_rot, config.get("w_rot")))
    if config.get("w_omega") is not None:
        expected_pairs.append(("w_omega", args.w_omega, config.get("w_omega")))

    if uses_aux_supervision(args.variant):
        if config.get("beta_aux") is not None:
            expected_pairs.append(("beta_aux", args.beta_aux, config.get("beta_aux")))
        if config.get("aux_cols") is not None:
            expected_pairs.append(("aux_cols", aux_cols, config.get("aux_cols")))

    if uses_lag(args.variant):
        expected_pairs.append(("lag_mode", args.lag_mode, config.get("lag_mode")))
        if config.get("alpha_init") is not None:
            expected_pairs.append(("alpha_init", args.alpha_init, config.get("alpha_init")))

    for name, current_value, saved_value in expected_pairs:
        if saved_value is None:
            continue
        if current_value != saved_value:
            mismatches.append(
                f"{name}: current={current_value!r}, checkpoint={saved_value!r}"
            )

    if mismatches:
        mismatch_text = "\n".join(f"  - {item}" for item in mismatches)
        raise ValueError(
            "Resume checkpoint does not match the current training configuration:\n"
            f"{mismatch_text}\n"
            "Please resume with the same arguments that created the checkpoint."
        )


def main():
    parser = argparse.ArgumentParser(description="Train unified PhysRes ablation variants")
    parser.add_argument("--train_trajs", type=str, default='["random","square","chirp"]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--variant", type=str, default="baseline", choices=VARIANT_CHOICES)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--lr-start", "--lr_start", dest="lr_start", type=float, default=1e-5)
    parser.add_argument("--lr-end", "--lr_end", dest="lr_end", type=float, default=1e-8)
    parser.add_argument("--loss-type", type=str, default="exp", choices=["exp", "mixed"])
    parser.add_argument("--beta_geo", type=float, default=0.01)
    parser.add_argument("--beta_aux", type=float, default=0.05)
    parser.add_argument("--beta-force", type=float, default=0.1)
    parser.add_argument("--w_rot", type=float, default=2.0)
    parser.add_argument("--w_omega", type=float, default=2.0)
    parser.add_argument("--lag_mode", type=str, default="per_motor", choices=["shared", "per_motor"])
    parser.add_argument("--alpha_init", type=float, default=0.85)
    parser.add_argument("--aux_cols", type=str, default='["az_body"]')
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA automatic mixed precision for faster training.",
    )
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_latest_every", type=int, default=1)
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Checkpoint path to resume from, or 'latest'/'best' for this run.",
    )
    parser.add_argument("--out_model_dir", type=str, default=str(PROJECT_ROOT / "out" / "models"))
    parser.add_argument("--scaler_root", type=str, default=str(PROJECT_ROOT / "scalers"))
    add_wandb_args(parser)
    args = parser.parse_args()

    if uses_force_supervision(args.variant) and args.loss_type != "mixed":
        print("⚠️ lag_gru_force uses mixed temporal loss. Overriding --loss-type to 'mixed'.")
        args.loss_type = "mixed"

    train_trajs = parse_json_list(args.train_trajs, "train_trajs")
    aux_cols = parse_json_list(args.aux_cols, "aux_cols")
    if uses_force_supervision(args.variant) and args.aux_cols == DEFAULT_AUX_COLS_RAW:
        aux_cols = DEFAULT_FORCE_AUX_COLS
    if uses_force_supervision(args.variant) and len(aux_cols) < 3:
        print(
            "⚠️ lag_gru_force expects 3 auxiliary body-acceleration channels. "
            "Force loss will be skipped unless aux_cols provides at least 3 values."
        )
    train_runs = [1, 2, 3]
    valid_runs = [4]
    valid_trajs = train_trajs
    batch_size = args.batch_size
    dt = 0.01
    lr_start = args.lr_start
    lr_end = args.lr_end
    seed = 42

    device_str = args.device
    if device_str.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    set_global_seed(seed)
    print(f"🌱 Global seed set to: {seed}")
    print(f"⚙️ Batch size: {batch_size} | AMP enabled: {amp_enabled}")

    model_name = build_model_name(args.variant, train_trajs)
    print(f"🧠 Model name composed automatically: {model_name}")

    model_dir = Path(args.out_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    latest_path = model_dir / f"{model_name}__latest.pt"
    print(f"✅ Model will be saved to: {model_path}")

    resume_checkpoint = None
    resume_path = resolve_checkpoint_path(args.resume_path, model_dir, model_name)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"❌ Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        validate_resume_checkpoint(
            checkpoint=resume_checkpoint,
            args=args,
            train_trajs=train_trajs,
            valid_trajs=valid_trajs,
            aux_cols=aux_cols,
        )
        print(f"🔁 Resuming from checkpoint: {resume_path.resolve()}")

    scaler_root = Path(args.scaler_root)
    if resume_checkpoint is not None and resume_checkpoint.get("scaler_dir") is not None:
        scaler_dir = Path(resume_checkpoint["scaler_dir"])
    else:
        scaler_dir = scaler_root / model_name
    scaler_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "train"
    use_aux = uses_aux_supervision(args.variant)
    use_aux_data = uses_aux_dataset(args.variant)
    force_aux_available = uses_force_supervision(args.variant)

    try:
        train_ds = load_split(
            train_trajs,
            train_runs,
            data_dir,
            "train",
            args.horizon,
            use_aux=use_aux_data,
            aux_cols=aux_cols,
        )
        valid_ds = load_split(
            valid_trajs,
            valid_runs,
            data_dir,
            "valid",
            args.horizon,
            use_aux=use_aux_data,
            aux_cols=aux_cols,
        )
    except Exception as exc:
        if not uses_force_supervision(args.variant):
            raise
        print(
            "⚠️ Force auxiliary labels are unavailable. "
            f"Continuing without force supervision: {exc}"
        )
        force_aux_available = False
        train_ds = load_split(
            train_trajs,
            train_runs,
            data_dir,
            "train",
            args.horizon,
            use_aux=False,
            aux_cols=aux_cols,
        )
        valid_ds = load_split(
            valid_trajs,
            valid_runs,
            data_dir,
            "valid",
            args.horizon,
            use_aux=False,
            aux_cols=aux_cols,
        )

    if use_aux:
        train_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(train_ds), scale=True, fold="train", scaler_dir=scaler_dir
        )
        valid_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(valid_ds), scale=True, fold="valid", scaler_dir=scaler_dir
        )
        resolved_aux_cols = getattr(train_ds[0], "aux_cols", aux_cols)
        aux_dim = train_dataset.aux_seq.shape[-1]
    elif force_aux_available:
        train_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(train_ds),
            scale=True,
            fold="train",
            scaler_dir=scaler_dir,
            scale_aux=False,
        )
        valid_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(valid_ds),
            scale=True,
            fold="valid",
            scaler_dir=scaler_dir,
            scale_aux=False,
        )
        resolved_aux_cols = getattr(train_ds[0], "aux_cols", aux_cols)
        aux_dim = train_dataset.aux_seq.shape[-1]
        if aux_dim < 3:
            print(
                "⚠️ Force auxiliary supervision is active but aux_dim < 3. "
                "Residual force loss will stay disabled."
            )
    else:
        train_dataset = combine_concat_dataset(
            ConcatDataset(train_ds), scale=True, fold="train", scaler_dir=scaler_dir
        )
        valid_dataset = combine_concat_dataset(
            ConcatDataset(valid_ds), scale=True, fold="valid", scaler_dir=scaler_dir
        )
        resolved_aux_cols = aux_cols
        aux_dim = len(aux_cols)

    traj_info = {"train_trajs": train_trajs, "valid_trajs": valid_trajs}
    traj_info_path = scaler_dir / "trajectories.json"
    with open(traj_info_path, "w") as handle:
        json.dump(traj_info, handle, indent=4)
    print(f"📝 Saved trajectory info to {traj_info_path}")

    train_generator = build_torch_generator(seed)
    valid_generator = build_torch_generator(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=valid_generator,
        worker_init_fn=seed_worker,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    phys_params = build_phys_params()
    model, hidden_dim, num_layers, residual_input_dim = build_model(
        variant=args.variant,
        phys_params=phys_params,
        dt=dt,
        x_scaler=train_dataset.x_scaler,
        u_scaler=train_dataset.u_scaler,
        lag_mode=args.lag_mode,
        alpha_init=args.alpha_init,
        aux_dim=aux_dim,
        hidden_dim=args.hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim,
    )
    model = model.to(device)
    model.beta_force = args.beta_force
    print(f"🧩 Initialized model variant: {args.variant}")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_trainable_params:,}")

    criterion = build_criterion(
        variant=args.variant,
        x_scaler=train_dataset.x_scaler,
        beta_geo=args.beta_geo,
        beta_aux=args.beta_aux,
        w_rot=args.w_rot,
        w_omega=args.w_omega,
    ) if not (uses_plain_temporal_loss(args.variant) and args.loss_type == "mixed") else None
    if criterion is not None:
        criterion = criterion.to(device)

    optimizer = build_optimizer(model, args.variant, lr_start)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr_end
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    wandb_run = init_wandb_run(
        args=args,
        project_root=PROJECT_ROOT,
        model_name=model_name,
        config={
            **vars(args),
            "model_type": "physres_ablation",
            "train_trajs": train_trajs,
            "valid_trajs": valid_trajs,
            "train_runs": train_runs,
            "valid_runs": valid_runs,
            "batch_size": batch_size,
            "seed": seed,
            "dt": dt,
            "device": str(device),
            "use_aux": use_aux,
            "force_aux_available": force_aux_available,
            "resolved_aux_cols": resolved_aux_cols,
            "aux_dim": aux_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "residual_input_dim": residual_input_dim,
            "model_path": model_path,
            "scaler_dir": scaler_dir,
            "num_trainable_params": num_trainable_params,
            "train_dataset_size": len(train_dataset),
            "valid_dataset_size": len(valid_dataset),
            "phys_params": phys_params,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "amp": amp_enabled,
        },
        tags=["physres-ablation", args.variant, *train_trajs],
        group=f"physres_ablation__{'_'.join(train_trajs)}",
    )
    maybe_watch_model(wandb_run, model, args)

    best_val_loss = float("inf")
    best_epoch = None
    start_epoch = 0

    if args.save_latest_every > 0:
        print(
            f"🔁 Latest resumable checkpoint will be updated every "
            f"{args.save_latest_every} epoch(s): {latest_path}"
        )

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])

        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
        else:
            print("⚠️ Resume checkpoint is missing optimizer state. Optimizer will restart.")

        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        else:
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            print(
                "⚠️ Resume checkpoint is missing scheduler state. "
                "Learning-rate schedule will restart from the current script settings."
            )

        best_val_loss = resume_checkpoint.get(
            "best_val_loss",
            resume_checkpoint.get("val_loss", float("inf")),
        )
        best_epoch = resume_checkpoint.get("best_epoch")
        start_epoch = max(resume_checkpoint.get("epoch", -1) + 1, 0)

        print(
            f"🔁 Restored epoch {resume_checkpoint.get('completed_epoch', start_epoch)} "
            f"and will continue from epoch {start_epoch + 1}/{args.epochs}"
        )
        if best_epoch is not None:
            print(
                f"🏁 Best checkpoint so far: epoch {best_epoch} "
                f"with valid loss {best_val_loss:.6f}"
            )

        if start_epoch >= args.epochs:
            print("✅ Resume checkpoint already reached the requested total epochs. Nothing to do.")
            finish_run(
                wandb_run,
                summary={
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_path": model_path,
                    "variant": args.variant,
                    "resume_path": str(resume_path.resolve()),
                },
            )
            return

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        train_metric_totals = init_metric_totals()
        train_grad_norm_total = 0.0
        train_grad_norm_max = 0.0
        train_generator.manual_seed(seed + epoch)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(device, amp_enabled):
                total_loss, loss_dict = compute_loss(
                    model,
                    criterion,
                    batch,
                    device,
                    args.variant,
                    args.loss_type,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = compute_gradient_norm(model)
            train_grad_norm_total += grad_norm
            train_grad_norm_max = max(train_grad_norm_max, grad_norm)
            scaler.step(optimizer)
            scaler.update()
            update_metric_totals(train_metric_totals, loss_dict)

        avg_train_metrics = average_metrics(train_metric_totals, len(train_loader))
        avg_train_grad_norm = train_grad_norm_total / len(train_loader)

        model.eval()
        valid_metric_totals = init_metric_totals()
        with torch.no_grad():
            for batch in valid_loader:
                with build_autocast_context(device, amp_enabled):
                    _, loss_dict = compute_loss(
                        model,
                        criterion,
                        batch,
                        device,
                        args.variant,
                        args.loss_type,
                    )
                update_metric_totals(valid_metric_totals, loss_dict)

        avg_valid_metrics = average_metrics(valid_metric_totals, len(valid_loader))
        current_lr = scheduler.get_last_lr()[0]
        is_best = avg_valid_metrics["loss_total"] < best_val_loss

        log_msg = (
            f"Epoch {epoch + 1}/{args.epochs} | LR={current_lr:.2e} | "
            f"Train={avg_train_metrics['loss_total']:.6f} | "
            f"Valid={avg_valid_metrics['loss_total']:.6f}"
        )
        if uses_geo_loss(args.variant):
            log_msg += (
                f" | TrainGeo={avg_train_metrics['loss_geo']:.6f}"
                f" | ValidGeo={avg_valid_metrics['loss_geo']:.6f}"
            )
        if uses_aux_supervision(args.variant):
            log_msg += (
                f" | TrainAux={avg_train_metrics['loss_aux']:.6f}"
                f" | ValidAux={avg_valid_metrics['loss_aux']:.6f}"
            )
        if "loss_exp" in avg_train_metrics:
            log_msg += (
                f" | TrainExp={avg_train_metrics['loss_exp']:.6f}"
                f" | ValidExp={avg_valid_metrics['loss_exp']:.6f}"
                f" | TrainTail={avg_train_metrics['loss_tail']:.6f}"
                f" | ValidTail={avg_valid_metrics['loss_tail']:.6f}"
            )
        if "loss_force" in avg_train_metrics:
            log_msg += (
                f" | TrainForce={avg_train_metrics['loss_force']:.6f}"
                f" | ValidForce={avg_valid_metrics['loss_force']:.6f}"
            )
        print(log_msg)

        if is_best:
            best_val_loss = avg_valid_metrics["loss_total"]
            best_epoch = epoch + 1

        scheduler.step()

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=avg_train_metrics,
            val_metrics=avg_valid_metrics,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            variant=args.variant,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            residual_input_dim=residual_input_dim,
            lag_mode=args.lag_mode,
            alpha_init=args.alpha_init,
            aux_cols=resolved_aux_cols,
            aux_dim=aux_dim,
            beta_geo=args.beta_geo,
            beta_aux=args.beta_aux,
            beta_force=args.beta_force,
            w_rot=args.w_rot,
            w_omega=args.w_omega,
            phys_params=phys_params,
            dt=dt,
            horizon=args.horizon,
            total_epochs=args.epochs,
            lr_start=lr_start,
            lr_end=lr_end,
            batch_size=batch_size,
            gru_hidden_dim=args.gru_hidden_dim,
            loss_type=args.loss_type,
            amp_enabled=amp_enabled,
            scaler_dir=scaler_dir,
            train_trajs=train_trajs,
            valid_trajs=valid_trajs,
            seed=seed,
        )

        if args.save_latest_every > 0 and (epoch + 1) % args.save_latest_every == 0:
            atomic_torch_save(checkpoint, latest_path)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_path = model_dir / f"{model_name}__epoch{epoch + 1:04d}.pt"
            atomic_torch_save(checkpoint, periodic_path)
            print(f"💾 Saved periodic checkpoint to {periodic_path}")

        if is_best:
            atomic_torch_save(checkpoint, model_path)
            print(
                f"💾 Saved best model at epoch {best_epoch} "
                f"with valid loss {avg_valid_metrics['loss_total']:.6f}"
            )

        wandb_metrics = {
            "epoch": epoch + 1,
            "optim/lr": current_lr,
            "train/grad_norm_mean": avg_train_grad_norm,
            "train/grad_norm_max": train_grad_norm_max,
            "model/param_norm": compute_parameter_norm(model),
            "best/val_loss": best_val_loss,
            "best/epoch": best_epoch or 0,
            "checkpoint/is_best": int(is_best),
            "timing/epoch_sec": time.time() - epoch_start,
        }
        wandb_metrics.update(prefix_metrics("train", avg_train_metrics))
        wandb_metrics.update(prefix_metrics("valid", avg_valid_metrics))
        wandb_metrics.update(collect_lag_metrics(model))
        log_metrics(wandb_run, wandb_metrics)

    print(f"✅ Training complete. Best model saved as {model_path} (epoch {best_epoch})")
    finish_run(
        wandb_run,
        summary={
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_path": model_path,
            "variant": args.variant,
        },
    )


if __name__ == "__main__":
    main()
