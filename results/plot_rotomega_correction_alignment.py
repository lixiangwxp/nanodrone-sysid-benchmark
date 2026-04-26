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

from results.model_comparison import require_csv
from utils.plot_utils import setup_matplotlib


ROTOMEGA_STATES = ["rx", "ry", "rz", "wx", "wy", "wz"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare model corrections against the oracle residual "
            "true_rotomega - pred_base_rotomega."
        )
    )
    parser.add_argument("--base-csv", required=True, type=str)
    parser.add_argument("--base-label", default="s000", type=str)
    parser.add_argument(
        "--variant-csv",
        action="append",
        dest="variant_csvs",
        default=[],
        help="Variant prediction CSV. Repeat once per variant.",
    )
    parser.add_argument(
        "--variant-label",
        action="append",
        dest="variant_labels",
        default=[],
        help="Variant label. Repeat in the same order as --variant-csv.",
    )
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--plot-dir", required=True, type=str)
    parser.add_argument(
        "--time-decimals",
        type=int,
        default=9,
        help="Decimals used to round start time before aligning CSV rows.",
    )
    parser.add_argument(
        "--align-mode",
        choices=["row", "time"],
        default="row",
        help=(
            "row aligns by CSV row index + horizon, best for same evaluation config; "
            "time aligns by rounded start time + horizon."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional CSV path for correction alignment metrics.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if not args.variant_csvs:
        raise ValueError("At least one --variant-csv is required.")
    if len(args.variant_csvs) != len(args.variant_labels):
        raise ValueError("--variant-label count must match --variant-csv count.")
    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be > 0.")


def require_columns(df, columns, source):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def build_rotomega_long_frame(df, max_horizon, time_decimals):
    require_columns(df, ["t", *ROTOMEGA_STATES], "prediction CSV")
    rows = []

    for horizon in range(1, max_horizon + 1):
        pred_cols = [f"{state}_pred_h{horizon}" for state in ROTOMEGA_STATES]
        require_columns(df, pred_cols, "prediction CSV")

        max_start = len(df) - horizon + 1
        if max_start <= 0:
            continue

        start_time = df["t"].iloc[:max_start].to_numpy(float)
        true_values = df[ROTOMEGA_STATES].shift(-(horizon - 1)).iloc[:max_start].to_numpy(float)
        pred_values = df[pred_cols].iloc[:max_start].to_numpy(float)

        finite_mask = (
            np.isfinite(start_time)
            & np.isfinite(true_values).all(axis=1)
            & np.isfinite(pred_values).all(axis=1)
        )
        if not finite_mask.any():
            continue

        frame = pd.DataFrame(
            {
                "row_index": np.arange(max_start, dtype=np.int64)[finite_mask],
                "start_time": start_time[finite_mask],
                "start_key": np.round(start_time[finite_mask], time_decimals),
                "horizon": horizon,
            }
        )
        for idx, state in enumerate(ROTOMEGA_STATES):
            frame[f"true_{state}"] = true_values[finite_mask, idx]
            frame[f"pred_{state}"] = pred_values[finite_mask, idx]
        rows.append(frame)

    if not rows:
        raise ValueError("No valid rot/omega prediction rows were built.")
    return pd.concat(rows, ignore_index=True)


def pearson_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return np.nan
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return np.nan
    return float(np.dot(a_centered, b_centered) / denom)


def cosine_similarity(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return np.nan
    return float(np.dot(a, b) / denom)


def magnitude_ratio(correction, oracle):
    denom = np.nanmean(np.abs(oracle))
    if not np.isfinite(denom) or denom == 0.0:
        return np.nan
    return float(np.nanmean(np.abs(correction)) / denom)


def compute_alignment_rows(base_frame, variant_frame, variant_label, align_mode):
    join_keys = ["horizon", "row_index"] if align_mode == "row" else ["horizon", "start_key"]
    merged = base_frame.merge(
        variant_frame,
        on=join_keys,
        suffixes=("_base", "_variant"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No aligned rows found for variant {variant_label!r}.")

    rows = []
    for state in ROTOMEGA_STATES:
        oracle = merged[f"true_{state}_base"].to_numpy(float) - merged[
            f"pred_{state}_base"
        ].to_numpy(float)
        correction = merged[f"pred_{state}_variant"].to_numpy(float) - merged[
            f"pred_{state}_base"
        ].to_numpy(float)
        rows.append(
            {
                "variant_label": variant_label,
                "channel": state,
                "align_mode": align_mode,
                "n": int(np.isfinite(oracle).sum()),
                "corr": pearson_corr(correction, oracle),
                "cosine_similarity": cosine_similarity(correction, oracle),
                "magnitude_ratio": magnitude_ratio(correction, oracle),
                "oracle_abs_mean": float(np.nanmean(np.abs(oracle))),
                "correction_abs_mean": float(np.nanmean(np.abs(correction))),
            }
        )
    return rows


def compute_summary(args):
    base_df = require_csv(Path(args.base_csv).expanduser().resolve())
    base_frame = build_rotomega_long_frame(base_df, args.max_horizon, args.time_decimals)

    rows = []
    for csv_path, label in zip(args.variant_csvs, args.variant_labels):
        variant_df = require_csv(Path(csv_path).expanduser().resolve())
        variant_frame = build_rotomega_long_frame(
            variant_df,
            args.max_horizon,
            args.time_decimals,
        )
        rows.extend(compute_alignment_rows(base_frame, variant_frame, label, args.align_mode))

    return pd.DataFrame(rows)


def plot_alignment_bars(summary_df, plot_dir):
    setup_matplotlib()
    metrics = ["corr", "cosine_similarity", "magnitude_ratio"]
    ylabels = ["Pearson corr", "Cosine similarity", "Magnitude ratio"]
    variants = list(summary_df["variant_label"].drop_duplicates())
    x = np.arange(len(ROTOMEGA_STATES))
    width = 0.8 / max(len(variants), 1)
    colors = plt.get_cmap("tab10").colors

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True, dpi=180)
    for metric_idx, metric in enumerate(metrics):
        ax = axs[metric_idx]
        for variant_idx, variant in enumerate(variants):
            group = summary_df[summary_df["variant_label"] == variant]
            values = [
                group.loc[group["channel"] == channel, metric].iloc[0]
                if (group["channel"] == channel).any()
                else np.nan
                for channel in ROTOMEGA_STATES
            ]
            offset = (variant_idx - (len(variants) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=variant,
                color=colors[variant_idx % len(colors)],
            )
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        if metric == "magnitude_ratio":
            ax.axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabels[metric_idx])
        ax.grid(True, axis="y", alpha=0.3)

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(ROTOMEGA_STATES)
    axs[-1].set_xlabel("Rot/omega channel")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 4))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "rotomega_correction_alignment.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    plot_dir = Path(args.plot_dir).expanduser().resolve()
    summary_df = compute_summary(args)
    plot_path = plot_alignment_bars(summary_df, plot_dir)

    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path is not None
        else plot_dir / "rotomega_correction_alignment.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
