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
    return variant in {"lag", "lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_histrotres", "lag_gru_actbank_alphaonly", "lag_gru_actbank_omegares", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force", "lag_geo", "full"}


def uses_hist_init(variant):
    return variant == "lag_gru_histinit_honly"


def uses_history(variant):
    return variant in {"lag_gru_histinit_honly", "lag_gru_actbank_alphaonly", "lag_gru_actbank_omegares", "lag_gru_histrotres"}


DEBUG_TRACE_VARIANTS = {
    "lag_gru",
    "lag_gru_uinit",
    "lag_gru_histinit_honly",
    "lag_gru_histrotres",
    "lag_gru_actbank_alphaonly",
    "lag_gru_actbank_omegares",
    "lag_gru_alpha4",
}


PRED_TRUE_TRACE_VARIANTS = {"baseline", "geo"}


def inverse_scaled_array(scaler, tensor):
    array = tensor.detach().cpu().numpy()
    if scaler is None:
        return array
    return scaler.inverse_transform(array.reshape(-1, array.shape[-1])).reshape(array.shape)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_debug_trace_batch(
    trace_path,
    *,
    pred_seq,
    true_seq,
    x0,
    u_seq,
    debug,
    x_scaler,
    u_scaler,
    dt,
    batch_index,
    sample_offset,
    start_indices=None,
    x_hist=None,
    u_hist=None,
    max_samples=64,
):
    sample_count = min(int(max_samples), int(pred_seq.shape[0]))
    if sample_count <= 0:
        return None

    sl = slice(0, sample_count)
    trace = {
        "batch_index": np.asarray(batch_index, dtype=np.int64),
        "sample_index": np.arange(sample_offset, sample_offset + sample_count, dtype=np.int64),
        "pred_seq": inverse_scaled_array(x_scaler, pred_seq[sl]),
        "true_seq": inverse_scaled_array(x_scaler, true_seq[sl]),
        "pred_seq_norm": tensor_to_numpy(pred_seq[sl]),
        "true_seq_norm": tensor_to_numpy(true_seq[sl]),
        "x0": inverse_scaled_array(x_scaler, x0[sl]),
        "u_seq": inverse_scaled_array(u_scaler, u_seq[sl]),
        "x0_norm": tensor_to_numpy(x0[sl]),
        "u_seq_norm": tensor_to_numpy(u_seq[sl]),
    }

    if start_indices is not None:
        start_index_np = tensor_to_numpy(start_indices[sl]).astype(np.int64)
        trace["start_index"] = start_index_np
        trace["t_start"] = start_index_np.astype(np.float64) * float(dt)
        trace["t"] = trace["t_start"][:, None] + (
            np.arange(1, pred_seq.shape[1] + 1, dtype=np.float64)[None, :] * float(dt)
        )

    if x_hist is not None:
        trace["x_hist"] = inverse_scaled_array(x_scaler, x_hist[sl])
        trace["x_hist_norm"] = tensor_to_numpy(x_hist[sl])
    if u_hist is not None:
        trace["u_hist"] = inverse_scaled_array(u_scaler, u_hist[sl])
        trace["u_hist_norm"] = tensor_to_numpy(u_hist[sl])

    for key, value in debug.items():
        trace[key] = tensor_to_numpy(value[sl])

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **trace)
    return trace_path


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


def load_test_datasets(test_trajs, data_root, horizon, history_len=0):
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
    if variant in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_histrotres", "lag_gru_actbank_alphaonly", "lag_gru_actbank_omegares", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"}:
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
    elif variant == "lag_gru_actbank_alphaonly":
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
            use_hist_init=False,
            hist_init_scale=0.1,
            use_actbank_alpha=True,
            actbank_use_history=True,
            actbank_taus_ms=tuple(
                config.get("actbank_taus_ms", [20.0, 50.0, 100.0, 200.0])
            ),
            actbank_alpha_scale=float(config.get("actbank_alpha_scale", 0.1)),
        ).to(device)
    elif variant == "lag_gru_actbank_omegares":
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
            use_hist_init=False,
            hist_init_scale=0.1,
            use_actbank_alpha=False,
            use_actbank_omegares=True,
            actbank_use_history=True,
            actbank_taus_ms=tuple(
                config.get("actbank_taus_ms", [20.0, 50.0, 100.0, 200.0])
            ),
            actbank_alpha_scale=float(config.get("actbank_alpha_scale", 0.1)),
            actbank_omegares_scale=float(config.get("actbank_omegares_scale", 0.05)),
            use_hist_rotres=False,
        ).to(device)
    elif variant == "lag_gru_histrotres":
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
            use_hist_init=False,
            hist_init_scale=0.1,
            use_actbank_alpha=False,
            actbank_use_history=False,
            use_hist_rotres=True,
            hist_rotres_scale=float(config.get("hist_rotres_scale", 0.02)),
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
    parser.add_argument("--test_trajs", type=str, default='["melon"]')
    parser.add_argument("--out_root", type=str, default=str(PROJECT_ROOT / "out" / "predictions"))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data" / "test"))
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--debug-trace-dir", type=str, default=None)
    parser.add_argument("--debug-trace-max-batches", type=int, default=1)
    parser.add_argument("--debug-trace-max-samples", type=int, default=64)
    args = parser.parse_args()
    if args.history_len < 0:
        raise ValueError("--history-len must be >= 0")
    if args.debug_trace_max_batches < 0:
        raise ValueError("--debug-trace-max-batches must be >= 0")
    if args.debug_trace_max_samples <= 0:
        raise ValueError("--debug-trace-max-samples must be > 0")

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
    supports_internal_debug = variant in DEBUG_TRACE_VARIANTS
    if args.debug_trace and variant not in DEBUG_TRACE_VARIANTS:
        print(
            "⚠️ Variant does not support internal debug variables; "
            "saving pred_seq/true_seq only."
        )
    history_len = checkpoint["config"].get("history_len", args.history_len)
    if uses_history(variant) and history_len <= 0:
        raise ValueError(
            f"{variant} requires history_len from checkpoint config "
            "or --history-len > 0"
        )

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
        history_len=history_len if uses_history(variant) else 0,
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
    needs_history = uses_history(variant)
    debug_trace_dir = None
    debug_saved_paths = []
    if args.debug_trace:
        if args.debug_trace_dir is None:
            debug_trace_dir = PROJECT_ROOT / "out" / "debug_traces" / model_name
        else:
            debug_trace_dir = Path(args.debug_trace_dir).expanduser()
        debug_trace_dir.mkdir(parents=True, exist_ok=True)

    sample_cursor = 0
    debug_batch_count = 0
    traj_label = "_".join(test_trajs)
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if len(batch) >= 5:
                x0, u_seq, x_seq_true, x_hist, u_hist = batch[:5]
            else:
                x0, u_seq, x_seq_true = batch[:3]
                x_hist = None
                u_hist = None
            if needs_history and x_hist is None:
                raise ValueError("Model requires history but batch does not provide x_hist/u_hist")
            x0_cpu = x0
            u_seq_cpu = u_seq
            x_seq_true_cpu = x_seq_true
            x_hist_cpu = x_hist
            u_hist_cpu = u_hist
            batch_size_actual = x0.shape[0]
            start_indices = None
            if getattr(test_dataset, "start_indices", None) is not None:
                start_indices = test_dataset.start_indices[
                    sample_cursor : sample_cursor + batch_size_actual
                ]

            x0 = x0.to(device)
            u_seq = u_seq.to(device)
            capture_debug = (
                args.debug_trace
                and debug_batch_count < args.debug_trace_max_batches
            )
            capture_internal_debug = capture_debug and supports_internal_debug

            if needs_history:
                x_hist = x_hist.to(device)
                u_hist = u_hist.to(device)
                if capture_internal_debug:
                    x_pred, debug = model(
                        x0,
                        u_seq,
                        x_hist=x_hist,
                        u_hist=u_hist,
                        return_debug=True,
                    )
                else:
                    x_pred = model(x0, u_seq, x_hist=x_hist, u_hist=u_hist)
                    debug = {}
            else:
                if capture_internal_debug:
                    x_pred, debug = model(x0, u_seq, return_debug=True)
                else:
                    x_pred = model(x0, u_seq)
                    debug = {}
            x_pred = x_pred.cpu()

            if capture_debug:
                trace_path = debug_trace_dir / f"{traj_label}_batch{debug_batch_count:03d}.npz"
                saved_path = save_debug_trace_batch(
                    trace_path,
                    pred_seq=x_pred,
                    true_seq=x_seq_true_cpu,
                    x0=x0_cpu,
                    u_seq=u_seq_cpu,
                    debug=debug,
                    x_scaler=test_dataset.x_scaler,
                    u_scaler=test_dataset.u_scaler,
                    dt=dt,
                    batch_index=batch_index,
                    sample_offset=sample_cursor,
                    start_indices=start_indices,
                    x_hist=x_hist_cpu if needs_history else None,
                    u_hist=u_hist_cpu if needs_history else None,
                    max_samples=args.debug_trace_max_samples,
                )
                if saved_path is not None:
                    debug_saved_paths.append(saved_path)
                    debug_batch_count += 1

            preds.append(x_pred)
            trues.append(x_seq_true)
            sample_cursor += batch_size_actual

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    x_scaler = test_dataset.x_scaler
    preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

    state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

    num_windows = preds.shape[0]
    start_index = history_len if uses_history(variant) else 0
    data = {"t": (np.arange(num_windows) + start_index) * dt}
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
    for debug_path in debug_saved_paths:
        print(f"🧪 Saved debug trace to: {debug_path}")

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
