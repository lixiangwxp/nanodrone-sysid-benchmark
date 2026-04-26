import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.model_comparison import add_rotation_columns, require_csv
from utils.metrics_utils import compute_errors
from utils.plot_utils import setup_matplotlib


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot rot/omega error curves over prediction horizon."
    )
    parser.add_argument(
        "--prediction-csv",
        action="append",
        dest="prediction_csvs",
        default=[],
        help="Prediction CSV path. Repeat once per model.",
    )
    parser.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        default=[],
        help="Model label. Repeat in the same order as --prediction-csv.",
    )
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--plot-dir", required=True, type=str)
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional CSV path for per-horizon rot/omega errors.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not args.prediction_csvs:
        raise ValueError("At least one --prediction-csv is required.")
    if len(args.prediction_csvs) != len(args.model_labels):
        raise ValueError("--model-label count must match --prediction-csv count.")
    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be > 0.")


def evaluate_models(args):
    rows = []
    for csv_path, label in zip(args.prediction_csvs, args.model_labels):
        df = add_rotation_columns(require_csv(Path(csv_path).expanduser().resolve()))
        metrics = compute_errors(df, args.max_horizon)
        for horizon in range(1, args.max_horizon + 1):
            rows.append(
                {
                    "model_label": label,
                    "horizon": horizon,
                    "rot_error": metrics["rot"][horizon],
                    "omega_error": metrics["omega"][horizon],
                }
            )
    return pd.DataFrame(rows)


def plot_error_curves(summary_df, plot_dir):
    setup_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharex=True, dpi=180)
    colors = plt.get_cmap("tab10").colors

    for idx, (label, group) in enumerate(summary_df.groupby("model_label", sort=False)):
        color = colors[idx % len(colors)]
        axs[0].plot(group["horizon"], group["rot_error"], label=label, color=color)
        axs[1].plot(group["horizon"], group["omega_error"], label=label, color=color)

    axs[0].set_ylabel("Rot error [rad]")
    axs[1].set_ylabel("Omega error [rad/s]")
    for ax in axs:
        ax.set_xlabel("Horizon step")
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 4))
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "rotomega_error_by_horizon.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    plot_dir = Path(args.plot_dir).expanduser().resolve()
    summary_df = evaluate_models(args)
    plot_path = plot_error_curves(summary_df, plot_dir)

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path is not None
        else plot_dir / "rotomega_error_by_horizon.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
