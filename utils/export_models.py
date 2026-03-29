import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from thop import profile
from thop.fx_profile import fx_profile

try:
    import torchinfo
except ImportError:  # pragma: no cover - optional dependency for pretty summaries
    torchinfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.models import PhysQuadModel, PhysResQuadModel, QuadLSTM, ResidualQuadModel


def infer_residual_config(state_dict):
    hidden_dim = state_dict["mlp.0.weight"].shape[0]
    state_dim = state_dict["out.weight"].shape[0]
    input_dim = state_dict["mlp.0.weight"].shape[1] - state_dim
    num_layers = len([k for k in state_dict if k.startswith("mlp.") and k.endswith(".weight")])
    return {
        "state_dim": state_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dt": 0.01,
    }


out_dir = PROJECT_ROOT / "out"

dt = 0.01
phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-2, 1e-2, 3e-3]),
}

device = "cpu"
Scaler = namedtuple("Scaler", ["mean_", "scale_"])

model_path = out_dir / "models" / "residual_random_square_chirp.pt"
ckpt = torch.load(model_path, map_location=device)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
cfg = ckpt.get("config", infer_residual_config(state)) if isinstance(ckpt, dict) else infer_residual_config(state)
residual_model = ResidualQuadModel(**cfg)
residual_model.load_state_dict(state)
residual_model.eval()

models = {
    "residual": residual_model,
    # "physical": PhysQuadModel(phys_params, dt),
    # "phys+residual": PhysResQuadModel(...),
    # "lstm": QuadLSTM(...),
}

for model_name, model in models.items():
    onnx_dir = out_dir / "export" / model_name
    onnx_dir.mkdir(parents=True, exist_ok=True)

    x0, u = (torch.zeros((1, 12)), torch.zeros((1, 1, 4)))

    macs, params = profile(model, inputs=(x0, u), verbose=True, report_missing=True)
    print(f"THOP profiling: {macs} MACS, {params} params")

    try:
        flops = fx_profile(model, input=(x0, u), verbose=False)
        print(f"THOP FX profiling: {flops} FLOPs")
    except Exception as exc:  # pragma: no cover - best effort profiler
        print("THOP FX profiling failed")
        print(exc)

    if torchinfo is not None:
        torchinfo.summary(
            model,
            input_data=(x0, u),
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ],
            depth=4,
        )
    else:
        print("ℹ️ torchinfo is not installed; skipping model summary.")

    onnx_path = onnx_dir / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        (x0, u),
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["x0", "u"],
        output_names=["x1"],
        training=torch.onnx.TrainingMode.TRAINING,
    )
