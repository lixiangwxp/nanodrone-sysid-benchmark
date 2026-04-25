import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MOTOR_LABELS = ["m1", "m2", "m3", "m4"]
PHYS_LABELS = ["T", "tau_x", "tau_y", "tau_z"]
ROTOMEGA_LABELS = ["rx", "ry", "rz", "wx", "wy", "wz"]


SUMMARY_COLUMNS = [
    "alpha_mean",
    "alpha_std",
    "alpha_min",
    "alpha_max",
    "alpha_low_frac",
    "alpha_high_frac",
    "u_diff_abs_mean",
    "u_diff_abs_max",
    "u_phys_T_diff_abs_mean",
    "u_phys_tau_diff_abs_mean",
    "dx_res_delta_posvel_norm_mean",
    "dx_res_delta_rotomega_norm_mean",
    "delta_alpha_abs_mean",
    "delta_rotomega_abs_mean",
]


def load_trace(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def sample_trace_array(trace, key, sample_index):
    if key not in trace:
        return None
    array = trace[key]
    if array.ndim == 0:
        return array
    if sample_index < 0 or sample_index >= array.shape[0]:
        raise IndexError(
            f"--sample-index {sample_index} is out of range for {key} "
            f"with batch size {array.shape[0]}"
        )
    return array[sample_index]


def as_2d_series(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    return array.reshape(array.shape[0], -1)


def save_line_plot(path, y, labels, ylabel):
    y = as_2d_series(y)
    h = np.arange(1, y.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=160)
    for idx in range(y.shape[1]):
        label = labels[idx] if idx < len(labels) else f"dim{idx}"
        ax.plot(h, y[:, idx], linewidth=1.4, label=label)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=min(y.shape[1], 4), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_alpha(trace, sample_index, plot_dir):
    alpha = sample_trace_array(trace, "alpha", sample_index)
    if alpha is not None:
        labels = MOTOR_LABELS if as_2d_series(alpha).shape[1] == 4 else ["alpha"]
        save_line_plot(plot_dir / "alpha_timeseries.png", alpha, labels, "alpha")

    alpha_logits = sample_trace_array(trace, "alpha_logits", sample_index)
    if alpha_logits is not None:
        labels = MOTOR_LABELS if as_2d_series(alpha_logits).shape[1] == 4 else ["alpha_logits"]
        save_line_plot(
            plot_dir / "alpha_logits_timeseries.png",
            alpha_logits,
            labels,
            "alpha logits",
        )

    delta_alpha_logits = sample_trace_array(trace, "delta_alpha_logits", sample_index)
    if delta_alpha_logits is not None:
        save_line_plot(
            plot_dir / "delta_alpha_logits.png",
            delta_alpha_logits,
            ["delta_alpha_logits"],
            "delta alpha logits",
        )


def plot_u_raw_vs_eff(trace, sample_index, plot_dir):
    raw = sample_trace_array(trace, "u_raw_norm", sample_index)
    eff = sample_trace_array(trace, "u_eff_norm", sample_index)
    if raw is None or eff is None:
        return

    h = np.arange(1, raw.shape[0] + 1)
    fig, axs = plt.subplots(2, 2, figsize=(9, 5), sharex=True, dpi=160)
    for motor_idx, ax in enumerate(axs.flat):
        ax.plot(h, raw[:, motor_idx], "k--", linewidth=1.2, label="raw")
        ax.plot(h, eff[:, motor_idx], linewidth=1.4, label="eff")
        ax.set_title(MOTOR_LABELS[motor_idx])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.supxlabel("Horizon step")
    fig.supylabel("Normalized motor input")
    fig.tight_layout()
    fig.savefig(plot_dir / "u_raw_vs_u_eff_norm.png", bbox_inches="tight")
    plt.close(fig)


def plot_bank_state(trace, sample_index, plot_dir):
    bank_state = sample_trace_array(trace, "bank_state", sample_index)
    if bank_state is None:
        return
    if bank_state.ndim != 3:
        return

    h = np.arange(1, bank_state.shape[0] + 1)
    tau_labels = (
        ["tau=20ms", "tau=50ms", "tau=100ms", "tau=200ms"]
        if bank_state.shape[-1] == 4
        else [f"bank{k}" for k in range(bank_state.shape[-1])]
    )
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True, dpi=160)
    for motor_idx, ax in enumerate(axs.flat):
        for bank_idx in range(bank_state.shape[-1]):
            ax.plot(
                h,
                bank_state[:, motor_idx, bank_idx],
                linewidth=1.2,
                label=tau_labels[bank_idx],
            )
        ax.set_title(MOTOR_LABELS[motor_idx])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.supxlabel("Horizon step")
    fig.supylabel("Actuator bank state")
    fig.tight_layout()
    fig.savefig(plot_dir / "bank_state_by_motor.png", bbox_inches="tight")
    plt.close(fig)


def plot_dx_res_group_norm(trace, sample_index, plot_dir):
    before = sample_trace_array(trace, "dx_res_before", sample_index)
    delta = sample_trace_array(trace, "dx_res_delta", sample_index)
    if before is None or delta is None:
        return

    before_posvel = np.linalg.norm(before[:, 0:6], axis=-1)
    before_rotomega = np.linalg.norm(before[:, 6:12], axis=-1)
    delta_posvel = np.linalg.norm(delta[:, 0:6], axis=-1)
    delta_rotomega = np.linalg.norm(delta[:, 6:12], axis=-1)
    y = np.stack(
        [before_posvel, before_rotomega, delta_posvel, delta_rotomega],
        axis=-1,
    )
    save_line_plot(
        plot_dir / "dx_res_group_norm.png",
        y,
        [
            "before_posvel",
            "before_rotomega",
            "delta_posvel",
            "delta_rotomega",
        ],
        "Group norm",
    )


def plot_rotomega_pred_true(trace, sample_index, plot_dir):
    pred = sample_trace_array(trace, "pred_seq", sample_index)
    true = sample_trace_array(trace, "true_seq", sample_index)
    if pred is None or true is None:
        return

    h = np.arange(1, pred.shape[0] + 1)
    fig, axs = plt.subplots(3, 2, figsize=(9, 7), sharex=True, dpi=160)
    for local_idx, ax in enumerate(axs.flat):
        state_idx = 6 + local_idx
        ax.plot(h, pred[:, state_idx], linewidth=1.3, label="pred")
        ax.plot(h, true[:, state_idx], "k--", linewidth=1.2, label="true")
        ax.set_title(ROTOMEGA_LABELS[local_idx])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.supxlabel("Horizon step")
    fig.supylabel("State value")
    fig.tight_layout()
    fig.savefig(plot_dir / "rotomega_pred_true.png", bbox_inches="tight")
    plt.close(fig)


def nan_value():
    return float("nan")


def abs_mean(trace, key):
    if key not in trace:
        return nan_value()
    return float(np.nanmean(np.abs(trace[key])))


def build_summary_row(trace):
    row = {column: nan_value() for column in SUMMARY_COLUMNS}

    if "alpha" in trace:
        alpha = trace["alpha"]
        row["alpha_mean"] = float(np.nanmean(alpha))
        row["alpha_std"] = float(np.nanstd(alpha))
        row["alpha_min"] = float(np.nanmin(alpha))
        row["alpha_max"] = float(np.nanmax(alpha))
        row["alpha_low_frac"] = float(np.nanmean(alpha < 0.05))
        row["alpha_high_frac"] = float(np.nanmean(alpha > 0.95))

    if "u_diff_norm" in trace:
        u_diff_abs = np.abs(trace["u_diff_norm"])
        row["u_diff_abs_mean"] = float(np.nanmean(u_diff_abs))
        row["u_diff_abs_max"] = float(np.nanmax(u_diff_abs))

    if "u_phys_diff" in trace:
        u_phys_diff_abs = np.abs(trace["u_phys_diff"])
        row["u_phys_T_diff_abs_mean"] = float(np.nanmean(u_phys_diff_abs[..., 0]))
        row["u_phys_tau_diff_abs_mean"] = float(np.nanmean(u_phys_diff_abs[..., 1:4]))

    if "dx_res_delta" in trace:
        delta = trace["dx_res_delta"]
        row["dx_res_delta_posvel_norm_mean"] = float(
            np.nanmean(np.linalg.norm(delta[..., 0:6], axis=-1))
        )
        row["dx_res_delta_rotomega_norm_mean"] = float(
            np.nanmean(np.linalg.norm(delta[..., 6:12], axis=-1))
        )

    row["delta_alpha_abs_mean"] = abs_mean(trace, "delta_alpha_logits")
    row["delta_rotomega_abs_mean"] = abs_mean(trace, "delta_rotomega")
    return row


def write_summary_csv(path, row):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot saved actuator debug traces")
    parser.add_argument("--trace-npz", required=True, type=str)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--plot-dir", required=True, type=str)
    parser.add_argument("--summary-path", type=str, default=None)
    args = parser.parse_args(argv)

    trace_path = Path(args.trace_npz).expanduser().resolve()
    plot_dir = Path(args.plot_dir).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path is not None
        else plot_dir / "debug_summary.csv"
    )

    trace = load_trace(trace_path)

    plot_alpha(trace, args.sample_index, plot_dir)
    plot_u_raw_vs_eff(trace, args.sample_index, plot_dir)

    u_diff = sample_trace_array(trace, "u_diff_norm", args.sample_index)
    if u_diff is not None:
        save_line_plot(plot_dir / "u_diff_norm.png", u_diff, MOTOR_LABELS, "u_eff - u_raw")

    u_phys_diff = sample_trace_array(trace, "u_phys_diff", args.sample_index)
    if u_phys_diff is not None:
        save_line_plot(
            plot_dir / "u_phys_diff.png",
            u_phys_diff,
            PHYS_LABELS,
            "Normalized physical input diff",
        )

    plot_bank_state(trace, args.sample_index, plot_dir)
    plot_dx_res_group_norm(trace, args.sample_index, plot_dir)

    delta_rotomega = sample_trace_array(trace, "delta_rotomega", args.sample_index)
    if delta_rotomega is not None:
        save_line_plot(
            plot_dir / "delta_rotomega.png",
            delta_rotomega,
            ROTOMEGA_LABELS,
            "Rot/omega correction",
        )

    plot_rotomega_pred_true(trace, args.sample_index, plot_dir)
    write_summary_csv(summary_path, build_summary_row(trace))

    print(f"Saved debug plots to: {plot_dir}")
    print(f"Saved debug summary to: {summary_path}")


if __name__ == "__main__":
    main()
