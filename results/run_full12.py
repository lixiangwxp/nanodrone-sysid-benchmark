import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_WANDB_PROJECT = "nanodrone-sysid-benchmark"


TRAIN_TRAJS_DEFAULT = ["random", "square", "chirp"]
TEST_TRAJS_DEFAULT = ["melon"]


EXPERIMENT_SPECS = [
    {
        "id": "residual_weighted_mse",
        "model_label": "Residual+WeightedMSE",
        "model_family": "Residual",
        "loss_name": "WeightedMSE",
        "type": "standard",
        "prefix": "residual",
        "train_script": "train/train_residual.py",
        "test_script": "test/test_residual.py",
    },
    {
        "id": "lstm_weighted_mse",
        "model_label": "LSTM+WeightedMSE",
        "model_family": "LSTM",
        "loss_name": "WeightedMSE",
        "type": "standard",
        "prefix": "lstm",
        "train_script": "train/train_lstm.py",
        "test_script": "test/test_lstm.py",
    },
    {
        "id": "physres_weighted_mse",
        "model_label": "PhysRes+WeightedMSE",
        "model_family": "PhysRes",
        "loss_name": "WeightedMSE",
        "type": "standard",
        "prefix": "phys+res",
        "train_script": "train/train_phys+res.py",
        "test_script": "test/test_phys+res.py",
    },
    {
        "id": "physres_mixed_temporal",
        "model_label": "PhysRes+MixedTemporal",
        "model_family": "PhysRes",
        "loss_name": "MixedTemporal",
        "type": "ablation",
        "variant": "baseline",
        "loss_type": "mixed",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "physres_composite_geo",
        "model_label": "PhysRes+CompositeGeo",
        "model_family": "PhysRes",
        "loss_name": "CompositeGeo",
        "type": "ablation",
        "variant": "geo",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_weighted_mse",
        "model_label": "LagPhysRes+WeightedMSE",
        "model_family": "LagPhysRes",
        "loss_name": "WeightedMSE",
        "type": "ablation",
        "variant": "lag",
        "loss_type": "exp",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_mixed_temporal",
        "model_label": "LagPhysRes+MixedTemporal",
        "model_family": "LagPhysRes",
        "loss_name": "MixedTemporal",
        "type": "ablation",
        "variant": "lag",
        "loss_type": "mixed",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_composite_geo",
        "model_label": "LagPhysRes+CompositeGeo",
        "model_family": "LagPhysRes",
        "loss_name": "CompositeGeo",
        "type": "ablation",
        "variant": "lag_geo",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_gru_weighted_mse",
        "model_label": "LagPhysResGRU+WeightedMSE",
        "model_family": "LagPhysResGRU",
        "loss_name": "WeightedMSE",
        "type": "ablation",
        "variant": "lag_gru",
        "loss_type": "exp",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_gru_mixed_temporal",
        "model_label": "LagPhysResGRU+MixedTemporal",
        "model_family": "LagPhysResGRU",
        "loss_name": "MixedTemporal",
        "type": "ablation",
        "variant": "lag_gru",
        "loss_type": "mixed",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_gru_force_mixed_temporal_force",
        "model_label": "LagPhysResGRUForce+MixedTemporalForce",
        "model_family": "LagPhysResGRUForce",
        "loss_name": "MixedTemporalForce",
        "type": "ablation",
        "variant": "lag_gru_force",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
    {
        "id": "lag_physres_composite_geo_aux",
        "model_label": "LagPhysRes+CompositeGeoAux",
        "model_family": "LagPhysRes",
        "loss_name": "CompositeGeoAux",
        "type": "ablation",
        "variant": "full",
        "train_script": "train/train_physres_ablation.py",
        "test_script": "test/test_physres_ablation.py",
    },
]


def parse_json_list(raw_value, arg_name):
    parsed = json.loads(raw_value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must decode to a list or string")
    return [str(item) for item in parsed]


def uses_plain_temporal_loss(variant):
    return variant in {"baseline", "lag", "lag_gru", "lag_gru_force"}


def build_standard_model_name(prefix, train_trajs, name_suffix):
    model_name = f"{prefix}_{'_'.join(train_trajs)}"
    if name_suffix:
        model_name = f"{model_name}_{name_suffix}"
    return model_name


def build_ablation_model_name(variant, train_trajs, loss_type, name_suffix):
    parts = ["physres_ablation", variant]
    if uses_plain_temporal_loss(variant):
        parts.append(loss_type)
    parts.append("_".join(train_trajs))
    model_name = "__".join(parts)
    if name_suffix:
        model_name = f"{model_name}_{name_suffix}"
    return model_name


def json_arg(values):
    return json.dumps(values, separators=(",", ":"))


def write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def stream_command(cmd, log_path):
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    command_text = shlex.join(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"▶️ Running: {command_text}")

    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {command_text}\n")
        log_handle.flush()

        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()
        log_handle.write(f"[exit-code] {return_code}\n")

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def build_train_command(spec, args, checkpoint_path, wandb_group, wandb_run_name, name_suffix):
    train_trajs_raw = json_arg(args.train_trajs)
    cmd = [
        sys.executable,
        spec["train_script"],
        "--train_trajs",
        train_trajs_raw,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--horizon",
        str(args.horizon),
        "--batch-size",
        str(args.train_batch_size),
        "--name-suffix",
        name_suffix,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        wandb_group,
        "--wandb-run-name",
        wandb_run_name,
    ]
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_dir:
        cmd.extend(["--wandb-dir", str(args.wandb_dir)])

    if spec["type"] == "ablation":
        cmd.extend(["--variant", spec["variant"]])
        if "loss_type" in spec:
            cmd.extend(["--loss-type", spec["loss_type"]])
        cmd.extend(["--out_model_dir", str(checkpoint_path.parent)])
    return cmd


def build_test_command(spec, args, checkpoint_path):
    test_trajs_raw = json_arg(args.test_trajs)
    if spec["type"] == "ablation":
        return [
            sys.executable,
            spec["test_script"],
            "--model_path",
            str(checkpoint_path),
            "--device",
            args.device,
            "--horizon",
            str(args.horizon),
            "--test_trajs",
            test_trajs_raw,
        ]

    return [
        sys.executable,
        spec["test_script"],
        "--model-path",
        str(checkpoint_path),
        "--test-trajs",
        test_trajs_raw,
        "--batch-size",
        str(args.test_batch_size),
        "--horizon",
        str(args.horizon),
    ]


def build_summary_command(manifest_path, summary_path, max_horizon, expected_count=None):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "results" / "model_comparison.py"),
        "--manifest-path",
        str(manifest_path),
        "--summary-path",
        str(summary_path),
        "--max-horizon",
        str(max_horizon),
    ]
    if expected_count is not None:
        cmd.extend(["--expected-count", str(expected_count)])
    return cmd


def refresh_summary(manifest, manifest_path, summary_path, max_horizon, summary_log_path, expected_count=None):
    manifest["summary_status"] = "running"
    write_manifest(manifest_path, manifest)
    summary_cmd = build_summary_command(
        manifest_path=manifest_path,
        summary_path=summary_path,
        max_horizon=max_horizon,
        expected_count=expected_count,
    )
    stream_command(summary_cmd, summary_log_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"❌ Expected summary file not found: {summary_path}")
    manifest["summary_status"] = "completed" if expected_count is not None else "partial"
    write_manifest(manifest_path, manifest)


def build_experiment_record(spec, args, name_suffix, stamp, wandb_group, logs_dir):
    if spec["type"] == "standard":
        model_name = build_standard_model_name(spec["prefix"], args.train_trajs, name_suffix)
    else:
        model_name = build_ablation_model_name(
            spec["variant"],
            args.train_trajs,
            spec.get("loss_type", "exp"),
            name_suffix,
        )

    checkpoint_path = PROJECT_ROOT / "out" / "models" / f"{model_name}.pt"
    prediction_path = (
        PROJECT_ROOT
        / "out"
        / "predictions"
        / f"{model_name}_model_multistep"
        / f"{'_'.join(args.test_trajs)}_multistep.csv"
    )
    wandb_run_name = f"{spec['model_label']}__{stamp}"
    train_cmd = build_train_command(spec, args, checkpoint_path, wandb_group, wandb_run_name, name_suffix)
    test_cmd = build_test_command(spec, args, checkpoint_path)

    return {
        "id": spec["id"],
        "model_label": spec["model_label"],
        "model_family": spec["model_family"],
        "loss_name": spec["loss_name"],
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "wandb_run_name": wandb_run_name,
        "train_command": shlex.join(train_cmd),
        "test_command": shlex.join(test_cmd),
        "log_path": str(logs_dir / f"{spec['id']}.log"),
        "status": "pending",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full set of 12 model+loss experiments")
    parser.add_argument("--stamp", type=str, default=None, help="Batch stamp used for logs, manifests, and suffixes")
    parser.add_argument("--name-suffix", type=str, default=None, help="Optional suffix passed to training runs")
    parser.add_argument("--train-trajs", type=str, default=json.dumps(TRAIN_TRAJS_DEFAULT))
    parser.add_argument("--test-trajs", type=str, default=json.dumps(TEST_TRAJS_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["auto", "disabled", "online", "offline"],
    )
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(PROJECT_ROOT / "out" / "benchmark_summary_full12.csv"),
    )
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of experiment ids to run",
    )
    raw_args = parser.parse_args()
    raw_args.train_trajs = parse_json_list(raw_args.train_trajs, "train_trajs")
    raw_args.test_trajs = parse_json_list(raw_args.test_trajs, "test_trajs")
    if raw_args.wandb_dir is not None:
        raw_args.wandb_dir = Path(raw_args.wandb_dir).expanduser().resolve()
    if raw_args.only:
        valid_ids = {spec["id"] for spec in EXPERIMENT_SPECS}
        invalid_ids = sorted(set(raw_args.only) - valid_ids)
        if invalid_ids:
            raise ValueError(f"Unknown experiment ids in --only: {invalid_ids}")
    if raw_args.horizon < 50:
        raise ValueError("--horizon must be at least 50 for the fixed *_h50 summary columns")
    return raw_args


def main():
    args = parse_args()
    stamp = args.stamp or f"full12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name_suffix = args.name_suffix or stamp
    selected_specs = [
        spec for spec in EXPERIMENT_SPECS
        if args.only is None or spec["id"] in set(args.only)
    ]
    if not selected_specs:
        raise RuntimeError("❌ No experiments selected")

    logs_dir = PROJECT_ROOT / "out" / "logs" / stamp
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else PROJECT_ROOT / "out" / "manifests" / f"{stamp}.json"
    )
    summary_path = Path(args.summary_path).expanduser().resolve()
    wandb_group = args.wandb_group or f"{stamp}__{'_'.join(args.train_trajs)}"

    manifest = {
        "stamp": stamp,
        "name_suffix": name_suffix,
        "train_trajs": args.train_trajs,
        "test_trajs": args.test_trajs,
        "device": args.device,
        "epochs": args.epochs,
        "horizon": args.horizon,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": wandb_group,
        "summary_path": str(summary_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": "running",
        "summary_status": "pending",
        "experiments": [
            build_experiment_record(spec, args, name_suffix, stamp, wandb_group, logs_dir)
            for spec in selected_specs
        ],
    }
    write_manifest(manifest_path, manifest)

    current_experiment = None
    summary_log_path = logs_dir / "summary.log"
    try:
        for experiment in manifest["experiments"]:
            current_experiment = experiment
            log_path = Path(experiment["log_path"])
            train_cmd = shlex.split(experiment["train_command"])
            test_cmd = shlex.split(experiment["test_command"])
            checkpoint_path = Path(experiment["checkpoint_path"])
            prediction_path = Path(experiment["prediction_path"])

            experiment["status"] = "training"
            write_manifest(manifest_path, manifest)
            stream_command(train_cmd, log_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"❌ Expected checkpoint not found: {checkpoint_path}")

            experiment["status"] = "testing"
            write_manifest(manifest_path, manifest)
            stream_command(test_cmd, log_path)
            if not prediction_path.exists():
                raise FileNotFoundError(f"❌ Expected prediction file not found: {prediction_path}")

            experiment["status"] = "completed"
            write_manifest(manifest_path, manifest)

            refresh_summary(
                manifest=manifest,
                manifest_path=manifest_path,
                summary_path=summary_path,
                max_horizon=args.horizon,
                summary_log_path=summary_log_path,
            )

        refresh_summary(
            manifest=manifest,
            manifest_path=manifest_path,
            summary_path=summary_path,
            max_horizon=args.horizon,
            summary_log_path=summary_log_path,
            expected_count=len(manifest["experiments"]),
        )
        manifest["overall_status"] = "completed"
        write_manifest(manifest_path, manifest)
        print(f"✅ Full-12 run completed. Manifest: {manifest_path}")
        print(f"✅ Summary saved to: {summary_path}")
    except Exception as exc:
        if current_experiment is not None and current_experiment.get("status") not in {"completed", "failed"}:
            current_experiment["status"] = "failed"
        manifest["overall_status"] = "failed"
        if manifest.get("summary_status") == "running":
            manifest["summary_status"] = "failed"
        manifest["error"] = str(exc)
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
