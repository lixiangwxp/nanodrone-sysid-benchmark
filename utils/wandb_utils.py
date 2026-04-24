import json
from argparse import Namespace
from pathlib import Path

import torch

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_WANDB_PROJECT = "nanodrone-sysid-benchmark"


def add_wandb_args(parser, default_project=DEFAULT_WANDB_PROJECT):
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="auto",
        choices=["auto", "disabled", "online", "offline"],
        help="Weights & Biases mode. 'auto' uses the library default when installed.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=default_project,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name. Defaults to the auto-generated model name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name for comparing related runs.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated or JSON list of W&B tags.",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default=None,
        help="Local directory for W&B files. Defaults to <project>/out/wandb.",
    )
    parser.add_argument(
        "--wandb-watch",
        action="store_true",
        help="Log parameter/gradient histograms with wandb.watch().",
    )
    parser.add_argument(
        "--wandb-watch-log",
        type=str,
        default="all",
        choices=["gradients", "parameters", "all"],
        help="What wandb.watch() should record.",
    )
    parser.add_argument(
        "--wandb-watch-freq",
        type=int,
        default=100,
        help="How often wandb.watch() logs histograms.",
    )


def _make_json_safe(value):
    if isinstance(value, Namespace):
        return _make_json_safe(vars(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_tags(raw_tags):
    if raw_tags is None:
        return []
    raw_text = str(raw_tags).strip()
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw_text.split(",") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [str(parsed)]
    return [str(item).strip() for item in parsed if str(item).strip()]


def init_wandb_run(args, project_root, model_name, config, tags=None, group=None):
    if getattr(args, "wandb_mode", "auto") == "disabled":
        return None
    if wandb is None:
        print("ℹ️ wandb is not installed in this environment. Skipping W&B logging.")
        return None

    wandb_dir = Path(args.wandb_dir) if getattr(args, "wandb_dir", None) else Path(project_root) / "out" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    merged_tags = list(tags or [])
    merged_tags.extend(_parse_tags(getattr(args, "wandb_tags", "")))
    merged_tags = list(dict.fromkeys(tag for tag in merged_tags if tag))

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name or model_name,
        "group": args.wandb_group or group,
        "config": _make_json_safe(config),
        "tags": merged_tags,
        "job_type": "train",
        "dir": str(wandb_dir),
        "reinit": False,
    }
    if getattr(args, "wandb_mode", "auto") != "auto":
        init_kwargs["mode"] = args.wandb_mode

    try:
        run = wandb.init(**init_kwargs)
        if run is None:
            return None
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        return run
    except Exception as exc:
        print(f"⚠️ Failed to initialize W&B. Continuing without it: {exc}")
        return None


def maybe_watch_model(run, model, args):
    if run is None or not getattr(args, "wandb_watch", False):
        return
    try:
        run.watch(model, log=args.wandb_watch_log, log_freq=max(1, args.wandb_watch_freq))
    except Exception as exc:
        print(f"⚠️ Failed to enable wandb.watch(). Continuing without histograms: {exc}")


def log_metrics(run, metrics):
    if run is None:
        return
    try:
        run.log(_make_json_safe(metrics))
    except Exception as exc:
        print(f"⚠️ Failed to log metrics to W&B: {exc}")


def finish_run(run, summary=None):
    if run is None:
        return
    try:
        for key, value in (summary or {}).items():
            run.summary[key] = _make_json_safe(value)
        run.finish()
    except Exception as exc:
        print(f"⚠️ Failed to finalize W&B run cleanly: {exc}")


def prefix_metrics(prefix, metrics):
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def compute_gradient_norm(model):
    # Logging-only diagnostic: keep this opt-in because it can still synchronize devices.
    total_sq_norm = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq_norm += grad_norm ** 2
    return total_sq_norm ** 0.5


def compute_parameter_norm(model):
    # Logging-only diagnostic: keep this opt-in because it scans every parameter tensor.
    total_sq_norm = 0.0
    for param in model.parameters():
        param_norm = param.detach().data.norm(2).item()
        total_sq_norm += param_norm ** 2
    return total_sq_norm ** 0.5


def collect_lag_metrics(model):
    # Logging-only diagnostic: keep this opt-in because it materializes CPU scalars for W&B.
    alpha = getattr(model, "alpha", None)
    if alpha is None or not torch.is_tensor(alpha):
        return {}

    alpha_flat = alpha.detach().view(-1).cpu()
    metrics = {
        "model/lag_alpha_mean": alpha_flat.mean().item(),
        "model/lag_alpha_min": alpha_flat.min().item(),
        "model/lag_alpha_max": alpha_flat.max().item(),
    }
    if alpha_flat.numel() == 1:
        metrics["model/lag_alpha_shared"] = alpha_flat.item()
    else:
        for idx, value in enumerate(alpha_flat.tolist()):
            metrics[f"model/lag_alpha_motor_{idx + 1}"] = value
    return metrics
