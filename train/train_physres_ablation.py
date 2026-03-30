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
from dataset.dataset_aux import QuadDatasetWithAux, combine_concat_dataset_with_aux
from models.models import PhysQuadModel, PhysResQuadModel, ResidualQuadModel
from models.models_lag import LagPhysResQuadModel
from train.losses import WeightedMSELoss
from train.losses_ext import CompositeAblationLoss
from utils.seed_utils import build_torch_generator, seed_worker, set_global_seed


VARIANT_CHOICES = ["baseline", "geo", "lag", "lag_geo", "full"]


def parse_json_list(raw_value, arg_name):
    parsed = json.loads(raw_value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must decode to a list or string")
    return parsed


def uses_lag(variant):
    return variant in {"lag", "lag_geo", "full"}


def uses_geo_loss(variant):
    return variant in {"geo", "lag_geo", "full"}


def uses_aux_supervision(variant):
    return variant == "full"


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


def build_model(variant, phys_params, dt, x_scaler, u_scaler, lag_mode, alpha_init, aux_dim):
    hidden_dim = 64
    num_layers = 5
    residual_input_dim = 12 if uses_lag(variant) else 4

    phys_model = PhysQuadModel(phys_params, dt)
    residual_model = ResidualQuadModel(
        input_dim=residual_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dt=dt,
    )

    if uses_lag(variant):
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


def build_criterion(variant, x_scaler, beta_geo, beta_aux):
    if variant in {"baseline", "lag"}:
        return WeightedMSELoss(lambda_=0.1)

    return CompositeAblationLoss(
        x_scaler=x_scaler,
        lambda_mse=0.1,
        use_geo=uses_geo_loss(variant),
        beta_geo=beta_geo,
        use_aux=uses_aux_supervision(variant),
        beta_aux=beta_aux,
        aux_loss_type="smooth_l1",
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


def compute_loss(model, criterion, batch, device, variant):
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


def update_metric_totals(metric_totals, loss_dict):
    for key in metric_totals:
        metric_totals[key] += loss_dict[key].item()


def build_checkpoint(
    model,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    variant,
    hidden_dim,
    num_layers,
    residual_input_dim,
    lag_mode,
    alpha_init,
    aux_cols,
    aux_dim,
    phys_params,
    dt,
    horizon,
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
            "residual_input_dim": residual_input_dim,
            "dt": dt,
            "horizon": horizon,
        },
        "phys_params": phys_params,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_metrics["loss_total"],
        "val_loss": val_metrics["loss_total"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_trajs": train_trajs,
        "valid_trajs": valid_trajs,
        "seed": seed,
        "model_name": f"physres_ablation__{variant}__{'_'.join(train_trajs)}",
        "scaler_dir": str(scaler_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train unified PhysRes ablation variants")
    parser.add_argument("--train_trajs", type=str, default='["random","square","chirp"]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--variant", type=str, default="baseline", choices=VARIANT_CHOICES)
    parser.add_argument("--beta_geo", type=float, default=0.1)
    parser.add_argument("--beta_aux", type=float, default=0.05)
    parser.add_argument("--lag_mode", type=str, default="per_motor", choices=["shared", "per_motor"])
    parser.add_argument("--alpha_init", type=float, default=0.85)
    parser.add_argument("--aux_cols", type=str, default='["az_body"]')
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--out_model_dir", type=str, default=str(PROJECT_ROOT / "out" / "models"))
    parser.add_argument("--scaler_root", type=str, default=str(PROJECT_ROOT / "scalers"))
    args = parser.parse_args()

    train_trajs = parse_json_list(args.train_trajs, "train_trajs")
    aux_cols = parse_json_list(args.aux_cols, "aux_cols")
    train_runs = [1, 2, 3]
    valid_runs = [4]
    valid_trajs = train_trajs
    batch_size = 256
    dt = 0.01
    lr_start = 1e-5
    lr_end = 1e-8
    seed = 42

    device_str = args.device
    if device_str.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    set_global_seed(seed)
    print(f"🌱 Global seed set to: {seed}")

    model_name = f"physres_ablation__{args.variant}__{'_'.join(train_trajs)}"
    print(f"🧠 Model name composed automatically: {model_name}")

    model_dir = Path(args.out_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    print(f"✅ Model will be saved to: {model_path}")

    scaler_root = Path(args.scaler_root)
    scaler_dir = scaler_root / model_name
    scaler_dir.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / "data" / "train"
    use_aux = uses_aux_supervision(args.variant)

    train_ds = load_split(
        train_trajs,
        train_runs,
        data_dir,
        "train",
        args.horizon,
        use_aux=use_aux,
        aux_cols=aux_cols,
    )
    valid_ds = load_split(
        valid_trajs,
        valid_runs,
        data_dir,
        "valid",
        args.horizon,
        use_aux=use_aux,
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
    )
    model = model.to(device)
    print(f"🧩 Initialized model variant: {args.variant}")

    criterion = build_criterion(
        variant=args.variant,
        x_scaler=train_dataset.x_scaler,
        beta_geo=args.beta_geo,
        beta_aux=args.beta_aux,
    )
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr_end
    )

    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(args.epochs):
        model.train()
        train_metric_totals = init_metric_totals()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            optimizer.zero_grad()
            total_loss, loss_dict = compute_loss(model, criterion, batch, device, args.variant)
            total_loss.backward()
            optimizer.step()
            update_metric_totals(train_metric_totals, loss_dict)

        avg_train_metrics = average_metrics(train_metric_totals, len(train_loader))

        model.eval()
        valid_metric_totals = init_metric_totals()
        with torch.no_grad():
            for batch in valid_loader:
                _, loss_dict = compute_loss(model, criterion, batch, device, args.variant)
                update_metric_totals(valid_metric_totals, loss_dict)

        avg_valid_metrics = average_metrics(valid_metric_totals, len(valid_loader))
        current_lr = scheduler.get_last_lr()[0]

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
        print(log_msg)

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=avg_train_metrics,
            val_metrics=avg_valid_metrics,
            variant=args.variant,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            residual_input_dim=residual_input_dim,
            lag_mode=args.lag_mode,
            alpha_init=args.alpha_init,
            aux_cols=resolved_aux_cols,
            aux_dim=aux_dim,
            phys_params=phys_params,
            dt=dt,
            horizon=args.horizon,
            scaler_dir=scaler_dir,
            train_trajs=train_trajs,
            valid_trajs=valid_trajs,
            seed=seed,
        )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            periodic_path = model_dir / f"{model_name}__epoch{epoch + 1:04d}.pt"
            torch.save(checkpoint, periodic_path)
            print(f"💾 Saved periodic checkpoint to {periodic_path}")

        if avg_valid_metrics["loss_total"] < best_val_loss:
            best_val_loss = avg_valid_metrics["loss_total"]
            best_epoch = epoch + 1
            torch.save(checkpoint, model_path)
            print(
                f"💾 Saved best model at epoch {best_epoch} "
                f"with valid loss {avg_valid_metrics['loss_total']:.6f}"
            )

        scheduler.step()

    print(f"✅ Training complete. Best model saved as {model_path} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
