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
    LagPhysResGRUControlModel,
    LagPhysResGRUForceModel,
    LagPhysResGRUModel,
    LagPhysResGRUTorqueModel,
    LagPhysResQuadModel,
)
from train.losses_ext import build_criterion, build_loss_config, compute_loss
from utils.early_stopping import get_wait_count, is_improvement, should_stop_early
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


VARIANT_CHOICES = ["baseline", "geo", "lag", "lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force", "lag_geo", "full"]
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
    return variant in {"lag", "lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force", "lag_geo", "full"}


def uses_geo_loss(variant):
    return variant in {"geo", "lag_geo", "full"}


def uses_aux_supervision(variant):
    return variant == "full"


def uses_force_supervision(variant):
    return variant == "lag_gru_force"


def uses_aux_dataset(variant):
    return uses_aux_supervision(variant) or uses_force_supervision(variant)


def uses_hist_init(variant):
    return variant == "lag_gru_histinit_honly"

def build_model_name(variant, train_trajs, loss_type="exp", name_suffix=""):
    parts = ["physres", variant]
    parts.append(loss_type)
    model_name = "__".join(parts)
    if name_suffix:
        model_name = f"{model_name}_{name_suffix}"
    return model_name

#物理主干的先验常数
def build_phys_params():
    return {
        "g": 9.81,
        "m": 0.045,
        "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
        "thrust_to_weight": 2.0,
        "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
    }


#  是“把某个数据划分里的多个 CSV 文件读进来，并转成 dataset 对象列表”的。
def load_split(
    trajs,
    runs,
    data_dir,
    split,
    horizon,
    use_aux=False,
    aux_cols=None,
    history_len=0,
    start_offset=0,
):
    datasets = []
    for traj in trajs:
        for run in runs:
            file_name = f"{traj}_20251017_run{run}.csv"
            file_path = data_dir / file_name
            df = pd.read_csv(file_path)

            if use_aux:
                ds = QuadDatasetWithAux(
                    df,
                    horizon=horizon,
                    aux_cols=aux_cols,
                    use_acc_aux=True,
                )
            else:
                ds = QuadDataset(
                    df,
                    horizon=horizon,
                    history_len=history_len,
                    start_offset=start_offset,
                )
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
    torque_scale_factor,
    control_ctx_dim,
    hist_init_scale,
):
    num_layers = 5
    if variant in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"}:
        residual_input_dim = gru_hidden_dim + 12
        #对 lag_gru 相关的模型，残差网络看的是：GRU 隐状态（gru_hidden_dim维）和原始控制相关的12维量（u_raw：4维，u_eff：4维，u_raw - u_eff：4维），合起来就是 gru_hidden_dim + 12 维
        #GRU 隐状态 h 可以理解成一个“学出来的记忆向量”，
        #形状是 (B, gru_hidden_dim)。它不是显式物理量，不对应某个直接可测的状态，而是专门用来保存“12维状态和当前控制里没写出来，但对后续预测有用的信息”。
    elif uses_lag(variant):
        residual_input_dim = 12
        #对 lag 但不带 GRU 的模型，残差网络看的是：u_raw：4维，u_eff：4维，u_raw - u_eff：4维，合起来就是 12 维
    else:
        residual_input_dim = 4
        #残差网络只看原始控制u，也就是4个电机量。


    #物理主干，负责按动力学积分一步
    phys_model = PhysQuadModel(phys_params, dt)

    #神经网络修正项，负责学 physics 没覆盖好的误差
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
    elif variant == "lag_gru_uinit":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
            alpha_dim=1,
            use_u_init=True,
            u_init_scale=0.05,
        )
    elif variant == "lag_gru_histinit_honly":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
            alpha_dim=1,
            use_u_init=False,
            u_init_scale=0.05,
            use_hist_init=True,
            hist_init_scale=hist_init_scale,
        )
    elif variant == "lag_gru_alpha4":
        model = LagPhysResGRUModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
            alpha_dim=4,
        )
    elif variant == "lag_gru_ctrl":
        model = LagPhysResGRUControlModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
            control_ctx_dim=control_ctx_dim,
        )
    elif variant == "lag_gru_torque":
        model = LagPhysResGRUTorqueModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
            lag_mode=lag_mode,
            alpha_init=alpha_init,
            hidden_dim=gru_hidden_dim,
            torque_scale_factor=torque_scale_factor,
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
        #这一支涵盖：lag，lag_geo，full
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
        #不带 lag 的基础版
        model = PhysResQuadModel(
            phys=phys_model,
            residual=residual_model,
            x_scaler=x_scaler,
            u_scaler=u_scaler,
        )

    return model, hidden_dim, num_layers, residual_input_dim
#return 只给“调用者后续还要用的结果”；最终模型本体，残差网络隐藏维度，残差网络层数，残差网络输入维度，取决于 variant





def init_metric_totals():
    return {
        "loss_total": 0.0,
        "loss_mse": 0.0,
        "loss_geo": 0.0,
        "loss_aux": 0.0,
    }



#这个函数是在每个 batch 结束后，把当前 batch 的 loss 记到账本里。
def update_metric_totals(metric_totals, loss_dict):
    for key, value in loss_dict.items():
        metric_totals.setdefault(key, 0.0)
        metric_totals[key] += value.item()


def update_metric_totals_async(metric_totals, loss_dict):
    # Performance-only: keep train metrics on-device until epoch end to avoid per-batch sync.
    with torch.no_grad():
        for key, value in loss_dict.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            v = value.detach().float()
            if key not in metric_totals:
                metric_totals[key] = torch.zeros(
                    (),
                    device=v.device,
                    dtype=torch.float32,
                )
            metric_totals[key].add_(v)


#把一个epoch里累计的各项loss总和，除以batch数，变成平均值。
def average_metrics(metric_totals, num_batches):
    return {key: value / num_batches for key, value in metric_totals.items()}


def average_metrics_async(metric_totals, num_batches):
    return {
        key: (value / num_batches).item()
        for key, value in metric_totals.items()
    }


#加快cuda计算速度
def build_autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=torch.float16)



#配置怎么更新参数
def build_optimizer(model, variant, base_lr):
    weight_decay = 1e-5
    if variant not in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"}:
        return optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    grouped_ids = set()

    #从某个子模块里收集所有还没被收过的可训练参数
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

    #“快学习率”的参数组
    fast_params = collect_params(getattr(model, "u_init_head", None))
    fast_params.extend(collect_params(getattr(model, "hist_h_head", None)))
    fast_params.extend(collect_params(getattr(model, "alpha_head", None)))

    #“中等学习率”的参数组
    medium_params = collect_params(getattr(model, "gru_cell", None))
    medium_params.extend(collect_params(getattr(model, "hist_encoder", None)))
    medium_params.extend(collect_params(getattr(model, "control_encoder", None)))
    medium_params.extend(collect_params(getattr(model, "ctx_h0_adapter", None)))
    medium_params.extend(collect_params(getattr(model, "ctx_step_adapter", None)))
    medium_params.extend(collect_params(getattr(model, "torque_head", None)))
    medium_params.extend(collect_params(getattr(model, "force_head", None)))
    medium_params.extend(collect_params(getattr(model, "residual", None)))
    medium_params.extend(collect_params(getattr(model, "h_init", None)))

   #把前面没归类进去的剩余可训练参数，全部归到“慢速/默认组”
    slow_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in grouped_ids
    ]



    #把前面没归类进去的剩余可训练参数，全部归到“慢速/默认组”
    param_groups = []
    if fast_params:
        param_groups.append({"params": fast_params, "lr": 3e-4, "weight_decay": weight_decay})
    if medium_params:
        param_groups.append({"params": medium_params, "lr": 1e-4, "weight_decay": weight_decay})
    if slow_params:
        param_groups.append({"params": slow_params, "lr": base_lr, "weight_decay": weight_decay})

    return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)

#把“恢复训练所需要的信息”和“复现实验所需要的信息”打包成一个字典。
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
    torque_scale_factor,
    control_ctx_dim,
    history_len,
    hist_init_scale,
    init_from,
    freeze_non_torque,
    amp_enabled,
    scaler_dir,
    train_trajs,
    valid_trajs,
    seed,
    name_suffix,
    early_stop_patience,
    early_stop_min_delta,
    early_stop_start_epoch,
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
            "gru_hidden_dim": (
                gru_hidden_dim if variant in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"} else None
            ),
            "use_hist_init": bool(getattr(model, "use_hist_init", False)),
            "history_len": history_len if uses_hist_init(variant) else 0,
            "hist_init_scale": hist_init_scale if uses_hist_init(variant) else None,
            "loss_type": loss_type,
            "torque_scale_factor": torque_scale_factor,
            "control_ctx_dim": control_ctx_dim,
            "init_from": init_from,
            "freeze_non_torque": freeze_non_torque,
            "amp": amp_enabled,
            "seed": seed,
            "name_suffix": name_suffix,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_start_epoch": early_stop_start_epoch,
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

        "model_name": build_model_name(
            variant,
            train_trajs,
            loss_type=loss_type,
            name_suffix=name_suffix,
        ),
        "scaler_dir": str(scaler_dir),
    }

#安全地保存 checkpoint，避免写文件写到一半时把正式文件弄坏。
#先偷偷写临时文件，写完再一把替换掉正式文件。
def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

#resume断点续训后用，把优化器内部保存的状态张量搬到指定设备上
def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

# 从一个已有 checkpoint 里加载模型权重，用来做初始化，对应微调，只加载 模型权重
def load_init_weights(model, raw_path):
    init_from_path = Path(raw_path).expanduser()
    if not init_from_path.exists():
        raise FileNotFoundError(f"❌ Initialization checkpoint not found: {init_from_path}")

    init_payload = torch.load(init_from_path, map_location="cpu")

    init_state = (
        init_payload["model_state"]
        if isinstance(init_payload, dict) and "model_state" in init_payload
        else init_payload
    )

    load_result = model.load_state_dict(init_state, strict=False)

    resolved_path = str(init_from_path.resolve())

    print(f"🔁 Initialized model weights from: {resolved_path}")
    #打印实际加载的是哪个文件，方便日志和排查。
    print(f"   missing_keys: {load_result.missing_keys}")
    #当前模型里需要，但 checkpoint 里没有的参数
    print(f"   unexpected_keys: {load_result.unexpected_keys}")
    #checkpoint 里有，但当前模型里不存在的参数
    return resolved_path

#一个负责冻结参数，除了 torque_head 相关参数以外，其它参数全部冻结。局部微调。
def freeze_non_torque_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("torque_head"))

#一个负责统计参数量，返回可训练参数量和总参数量
def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    #表示模型里所有参数总数，不管冻没冻结。
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    #只统计 requires_grad=True 的参数，也就是训练时真的会更新的参数。
    return trainable_params, total_params

#一个负责把 resume 输入解析成真正的 checkpoint 路径
def resolve_checkpoint_path(raw_path, model_dir, model_name):
    if raw_path is None:
        return None
    if raw_path == "latest":
        return model_dir / f"{model_name}__latest.pt"
    if raw_path == "best":
        return model_dir / f"{model_name}.pt"
    return Path(raw_path).expanduser()

#防止你拿“best model 文件”去做断点续训。
def ensure_resume_checkpoint_is_resumable(resume_path, best_model_path):
    if Path(resume_path).resolve() == Path(best_model_path).resolve():
        raise ValueError(
            "The best-model checkpoint is not a resumable checkpoint because it does "
            "not preserve optimizer, scheduler, and early-stopping history beyond the "
            "epoch where the best model was found. Use --resume_path latest or an "
            "explicit latest/periodic checkpoint path instead."
        )

#断点续训前的配置一致性检查器
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
    if config.get("early_stop_patience") is not None:
        expected_pairs.append(
            ("early_stop_patience", args.early_stop_patience, config.get("early_stop_patience"))
        )
    if config.get("early_stop_min_delta") is not None:
        expected_pairs.append(
            (
                "early_stop_min_delta",
                args.early_stop_min_delta,
                config.get("early_stop_min_delta"),
            )
        )
    if config.get("early_stop_start_epoch") is not None:
        expected_pairs.append(
            (
                "early_stop_start_epoch",
                args.early_stop_start_epoch,
                config.get("early_stop_start_epoch"),
            )
        )

    if config.get("loss_type") is not None:
        expected_pairs.append(("loss_type", args.loss_type, config.get("loss_type")))
    if config.get("lr_start") is not None:
        expected_pairs.append(("lr_start", args.lr_start, config.get("lr_start")))
    if config.get("lr_end") is not None:
        expected_pairs.append(("lr_end", args.lr_end, config.get("lr_end")))
    if config.get("hidden_dim") is not None:
        expected_pairs.append(("hidden_dim", args.hidden_dim, config.get("hidden_dim")))
    if args.variant in {"lag_gru", "lag_gru_uinit", "lag_gru_histinit_honly", "lag_gru_alpha4", "lag_gru_ctrl", "lag_gru_torque", "lag_gru_force"} and config.get("gru_hidden_dim") is not None:
        expected_pairs.append(("gru_hidden_dim", args.gru_hidden_dim, config.get("gru_hidden_dim")))
    if args.variant == "lag_gru_histinit_honly":
        expected_pairs.append(("history_len", args.history_len, config.get("history_len", 0)))
        expected_pairs.append(
            (
                "hist_init_scale",
                args.hist_init_scale,
                config.get("hist_init_scale", 0.1),
            )
        )
    if args.variant == "lag_gru_ctrl":
        expected_pairs.append(
            (
                "control_ctx_dim",
                args.control_ctx_dim,
                config.get("control_ctx_dim", 32),
            )
        )
    if args.variant == "lag_gru_torque":
        expected_pairs.append(
            (
                "torque_scale_factor",
                args.torque_scale_factor,
                config.get("torque_scale_factor", 0.2),
            )
        )
        expected_pairs.append(
            (
                "freeze_non_torque",
                args.freeze_non_torque,
                config.get("freeze_non_torque", False),
            )
        )

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

#训练主函数
def main():
    parser = argparse.ArgumentParser(description="Train unified PhysRes ablation variants")
    parser.add_argument("--train_trajs", type=str, default='["random","square","chirp"]')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many epochs without validation improvement; 0 disables early stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum validation-loss improvement required to reset early stopping")
    parser.add_argument("--early-stop-start-epoch", type=int, default=0, help="Do not evaluate early stopping before this epoch number (1-indexed)")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default="baseline", choices=VARIANT_CHOICES)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument("--hist-init-scale", type=float, default=0.1)
    parser.add_argument("--lr-start", "--lr_start", dest="lr_start", type=float, default=1e-5)
    parser.add_argument("--lr-end", "--lr_end", dest="lr_end", type=float, default=1e-8)
    parser.add_argument("--loss-type", type=str, default="exp", choices=["exp", "mixed", "mixed_early"])
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the auto-generated model name",
    )
    parser.add_argument("--beta_geo", type=float, default=0.01)
    parser.add_argument("--beta_aux", type=float, default=0.05)
    parser.add_argument("--beta-force", type=float, default=0.1)
    parser.add_argument("--w_rot", type=float, default=2.0)
    parser.add_argument("--w_omega", type=float, default=2.0)
    parser.add_argument("--lag_mode", type=str, default="per_motor", choices=["shared", "per_motor"])
    parser.add_argument("--alpha_init", type=float, default=0.85)
    parser.add_argument(
        "--control-ctx-dim",
        type=int,
        default=32,
        help="Control context dimension for lag_gru_ctrl.",
    )
    parser.add_argument(
        "--torque-scale-factor",
        type=float,
        default=0.2,
        help="Scale factor for lag_gru_torque residual torque, multiplied by phys.max_torque.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Optional checkpoint path used only to partially initialize model weights before training.",
    )
    parser.add_argument(
        "--freeze-non-torque",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For lag_gru_torque only: freeze all parameters except torque_head.",
    )
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
    parser.add_argument(
        "--log-diagnostics-every",
        type=int,
        default=0,
        help="Compute expensive diagnostic norms every N training batches; 0 disables them.",
    )
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_latest_every", type=int, default=1)
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Resumable checkpoint path to continue training from, or 'latest' for this run.",
    )
    parser.add_argument("--out_model_dir", type=str, default=str(PROJECT_ROOT / "out" / "models"))
    parser.add_argument("--scaler_root", type=str, default=str(PROJECT_ROOT / "scalers"))
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.init_from is not None and args.resume_path is not None:
        raise ValueError("--init-from cannot be used together with --resume_path.")
    if args.freeze_non_torque and args.variant != "lag_gru_torque":
        raise ValueError("--freeze-non-torque is only supported for --variant lag_gru_torque.")
    if args.log_diagnostics_every < 0:
        raise ValueError("--log-diagnostics-every must be >= 0.")
    if args.history_len < 0:
        raise ValueError("--history-len must be >= 0.")
    if args.variant == "lag_gru_histinit_honly" and args.history_len <= 0:
        raise ValueError("--history-len must be > 0 for lag_gru_histinit_honly")

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
    seed = args.seed

    device_str = args.device
    if device_str.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    #设置随机种子
    set_global_seed(seed)
    print(f"🌱 Global seed set to: {seed}")
    print(f"⚙️ Batch size: {batch_size} | AMP enabled: {amp_enabled}")

    #自动生成模型名，根据 variant 和训练轨迹组合成一个有意义的名字，方便区分不同实验。
    model_name = build_model_name(
        args.variant,
        train_trajs,
        loss_type=args.loss_type,
        name_suffix=args.name_suffix,
    )
    print(f"🧠 Model name composed automatically: {model_name}")

    #准备模型输出目录
    model_dir = Path(args.out_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    #确定两个关键文件路径
    model_path = model_dir / f"{model_name}.pt"
    latest_path = model_dir / f"{model_name}__latest.pt"
    print(f"✅ Model will be saved to: {model_path}")


    #如果用户想断点续训，就先把 resume checkpoint 读进来并检查是否合法
    resume_checkpoint = None
    resume_path = resolve_checkpoint_path(args.resume_path, model_dir, model_name)
    if resume_path is not None:
        ensure_resume_checkpoint_is_resumable(resume_path, model_path)
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


    #确定 scaler 目录，存放数据标准化器
    scaler_root = Path(args.scaler_root)
    if resume_checkpoint is not None and resume_checkpoint.get("scaler_dir") is not None:
        scaler_dir = Path(resume_checkpoint["scaler_dir"])
    else:
        scaler_dir = scaler_root / model_name
    scaler_dir.mkdir(parents=True, exist_ok=True)


    #指定训练数据目录
    data_dir = PROJECT_ROOT / "data" / "train"

    #根据当前 variant 决定是否需要 aux 数据
    use_aux = uses_aux_supervision(args.variant)#要不要 aux loss
    use_aux_data = uses_aux_dataset(args.variant)#数据集要不要多读 aux 列
    force_aux_available = uses_force_supervision(args.variant)#：当前是不是 force-aware 这条分支
    history_len = args.history_len if uses_hist_init(args.variant) else 0

    try:
        train_ds = load_split(
            train_trajs,
            train_runs,
            data_dir,
            "train",
            args.horizon,
            use_aux=use_aux_data,
            aux_cols=aux_cols,
            history_len=history_len,
        )
        valid_ds = load_split(
            valid_trajs,
            valid_runs,
            data_dir,
            "valid",
            args.horizon,
            use_aux=use_aux_data,
            aux_cols=aux_cols,
            history_len=history_len,
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
            history_len=history_len,
        )
        valid_ds = load_split(
            valid_trajs,
            valid_runs,
            data_dir,
            "valid",
            args.horizon,
            use_aux=False,
            aux_cols=aux_cols,
            history_len=history_len,
        )

    if use_aux:
    #当前 variant 真的要做 aux supervision
        train_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(train_ds), scale=True, fold="train", scaler_dir=scaler_dir
        )
        valid_dataset = combine_concat_dataset_with_aux(
            ConcatDataset(valid_ds), scale=True, fold="valid", scaler_dir=scaler_dir
        )
        resolved_aux_cols = getattr(train_ds[0], "aux_cols", aux_cols)
        aux_dim = train_dataset.aux_seq.shape[-1]
    elif force_aux_available:
    #当前不是普通 aux supervision，但 force supervision 还可用。
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


    #判断 force loss 这次到底能不能开启。条件是：当前 variant 是 force-aware 的分支，并且 aux 数据里至少有 3 个维度可用来监督 xddot。
    force_loss_enabled = force_aux_available and aux_dim >= 3

    #把这次训练要用的 loss 配好，比如要不要 geo/aux/force，以及 loss_type 是 exp 还是 mixed。
    loss_config = build_loss_config(
        loss_type=args.loss_type,
        beta_geo=args.beta_geo,
        beta_aux=args.beta_aux,
        beta_force=args.beta_force,
        w_rot=args.w_rot,
        w_omega=args.w_omega,
        use_geo=uses_geo_loss(args.variant),
        use_aux=uses_aux_supervision(args.variant),
        use_force=uses_force_supervision(args.variant) and force_loss_enabled,
    )
    #保存 trajectories.json：把这次 train/valid 用了哪些轨迹记下来，方便复现。
    traj_info = {"train_trajs": train_trajs, "valid_trajs": valid_trajs}
    traj_info_path = scaler_dir / "trajectories.json"
    with open(traj_info_path, "w") as handle:
        json.dump(traj_info, handle, indent=4)
    print(f"📝 Saved trajectory info to {traj_info_path}")



    '''模型和训练器准备 '''
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

    phys_params = build_phys_params()#准备物理模型常数
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
        torque_scale_factor=args.torque_scale_factor,
        control_ctx_dim=args.control_ctx_dim,
        hist_init_scale=args.hist_init_scale,
        )#准备物理模型常数。
    model = model.to(device)
    resolved_init_from = None

    if args.init_from is not None:
        resolved_init_from = load_init_weights(model, args.init_from)

    if args.freeze_non_torque:
        freeze_non_torque_parameters(model)
        print("🔒 freeze_non_torque enabled: training torque_head parameters only.")
    print(f"🧩 Initialized model variant: {args.variant}")
    num_trainable_params, num_total_params = count_parameters(model)
    print(f"Trainable parameters: {num_trainable_params:,} / {num_total_params:,}")


    #构建损失函数
    criterion = build_criterion(
        loss_config=loss_config,
        x_scaler=train_dataset.x_scaler,
    ).to(device)

    #构建优化器
    optimizer = build_optimizer(model, args.variant, lr_start)

    #构建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr_end
    )

    #给 AMP 混合精度训练用。
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    wandb_run = init_wandb_run(
        args=args,
        project_root=PROJECT_ROOT,
        model_name=model_name,
        config={
            **vars(args),
            "model_type": "physres_ablation",
            "seed": seed,
            "use_aux": use_aux,
            "use_geo_loss": loss_config.use_geo,
            "use_aux_loss": loss_config.use_aux,
            "use_force_loss": loss_config.use_force,
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
            "control_ctx_dim": args.control_ctx_dim,
            "history_len": history_len,
            "hist_init_scale": args.hist_init_scale,
            "resolved_init_from": resolved_init_from,
        },
        tags=["physres-ablation", args.variant, *train_trajs],
        group=f"physres_ablation__{'_'.join(train_trajs)}",
    )
    maybe_watch_model(wandb_run, model, args)
    diagnostics_enabled = wandb_run is not None and args.log_diagnostics_every > 0



    best_val_loss = float("inf")
    best_epoch = None
    start_epoch = 0
    stopped_early = False

    #断点续训
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        #恢复优化器状态
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
        else:
            print("⚠️ Resume checkpoint is missing optimizer state. Optimizer will restart.")
        #恢复 scheduler 状态
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
                    "stopped_early": False,
                },
            )
            return

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        train_metric_totals = {}
        train_grad_norm_total = 0.0
        train_grad_norm_max = 0.0
        train_grad_norm_batches = 0
        train_generator.manual_seed(seed + epoch)
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            with build_autocast_context(device, amp_enabled):
                total_loss, loss_dict = compute_loss(
                    model,
                    criterion,
                    batch,
                    device,
                    loss_config,
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if diagnostics_enabled and batch_idx % args.log_diagnostics_every == 0:
                # Logging-only diagnostics: keep training math unchanged and run them sparsely.
                grad_norm = compute_gradient_norm(model)
                train_grad_norm_total += grad_norm
                train_grad_norm_max = max(train_grad_norm_max, grad_norm)
                train_grad_norm_batches += 1

            scaler.step(optimizer)
            scaler.update()
            update_metric_totals_async(train_metric_totals, loss_dict)

        avg_train_metrics = average_metrics_async(train_metric_totals, len(train_loader))
        avg_train_grad_norm = (
            train_grad_norm_total / train_grad_norm_batches
            if train_grad_norm_batches > 0
            else 0.0
        )

        #验证阶段：遍历 valid_loader
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
                        loss_config,
                    )
                update_metric_totals(valid_metric_totals, loss_dict)

        avg_valid_metrics = average_metrics(valid_metric_totals, len(valid_loader))
        current_lr = scheduler.get_last_lr()[0]
        is_best = is_improvement(
            avg_valid_metrics["loss_total"],
            best_val_loss,
            args.early_stop_min_delta,
        )
        early_stop_wait = get_wait_count(
            epoch + 1,
            best_epoch,
            start_epoch=args.early_stop_start_epoch,
        )
        early_stop_triggered = False

        log_msg = (
            f"Epoch {epoch + 1}/{args.epochs} | LR={current_lr:.2e} | "
            f"Train={avg_train_metrics['loss_total']:.6f} | "
            f"Valid={avg_valid_metrics['loss_total']:.6f}"
        )
        if loss_config.use_geo:
            log_msg += (
                f" | TrainGeo={avg_train_metrics['loss_geo']:.6f}"
                f" | ValidGeo={avg_valid_metrics['loss_geo']:.6f}"
            )
        if loss_config.use_aux:
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
            early_stop_wait = 0
        else:
            early_stop_wait = get_wait_count(
                epoch + 1,
                best_epoch,
                start_epoch=args.early_stop_start_epoch,
            )
            early_stop_triggered = should_stop_early(
                current_epoch=epoch + 1,
                best_epoch=best_epoch,
                start_epoch=args.early_stop_start_epoch,
                patience=args.early_stop_patience,
            )
            if early_stop_triggered:
                stopped_early = True
                print(
                    f"⏹️ Early stopping triggered at epoch {epoch + 1}/{args.epochs} "
                    f"after {early_stop_wait} epoch(s) without sufficient validation improvement. "
                    f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}"
                )

        if not early_stop_triggered:
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
            torque_scale_factor=args.torque_scale_factor,
            control_ctx_dim=args.control_ctx_dim,
            history_len=history_len,
            hist_init_scale=args.hist_init_scale,
            init_from=resolved_init_from,
            freeze_non_torque=args.freeze_non_torque,
            amp_enabled=amp_enabled,
            scaler_dir=scaler_dir,
            train_trajs=train_trajs,
            valid_trajs=valid_trajs,
            seed=seed,
            name_suffix=args.name_suffix,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_start_epoch=args.early_stop_start_epoch,
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
            "best/val_loss": best_val_loss,
            "best/epoch": best_epoch or 0,
            "checkpoint/is_best": int(is_best),
            "early_stop/wait_count": early_stop_wait,
            "early_stop/triggered": int(early_stop_triggered),
            "timing/epoch_sec": time.time() - epoch_start,
        }
        wandb_metrics.update(prefix_metrics("train", avg_train_metrics))
        wandb_metrics.update(prefix_metrics("valid", avg_valid_metrics))
        if diagnostics_enabled:
            wandb_metrics["model/param_norm"] = compute_parameter_norm(model)
            wandb_metrics.update(collect_lag_metrics(model))
        log_metrics(wandb_run, wandb_metrics)

        if early_stop_triggered:
            break

    if stopped_early:
        print(f"✅ Training stopped early. Best model saved as {model_path} (epoch {best_epoch})")
    else:
        print(f"✅ Training complete. Best model saved as {model_path} (epoch {best_epoch})")
    finish_run(
        wandb_run,
        summary={
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_path": model_path,
            "variant": args.variant,
            "stopped_early": stopped_early,
        },
    )


if __name__ == "__main__":
    main()
