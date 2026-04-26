import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.model_comparison import require_csv
from utils.plot_utils import setup_matplotlib


STATE_COLUMNS = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]
RAW_STATE_COLUMNS = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "qx",
    "qy",
    "qz",
    "qw",
    "wx",
    "wy",
    "wz",
]
OMEGA_STATES = ["wx", "wy", "wz"]
MOTOR_COLUMNS = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]
MODE_COLUMNS = ["T", "tau_x", "tau_y", "tau_z"]

ARM_LENGTH = 0.0353
KT = 3.72e-08
KC = 7.74e-12
MASS = 0.045
GRAVITY = 9.81
THRUST_TO_WEIGHT = 2.0
T_MAX = THRUST_TO_WEIGHT * MASS * GRAVITY
MAX_TORQUE = np.array([1e-2, 1e-2, 3e-3], dtype=float)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Offline diagnosis C: test whether baseline omega oracle residuals "
            "can be explained from causal history features."
        )
    )
    parser.add_argument(
        "--prediction-csv",
        action="append",
        dest="prediction_csvs",
        default=[],
        help="Baseline prediction CSV. Repeat once per trajectory label.",
    )
    parser.add_argument(
        "--traj-label",
        action="append",
        dest="traj_labels",
        default=[],
        help="Trajectory label for the corresponding --prediction-csv.",
    )
    parser.add_argument(
        "--raw-traj",
        action="append",
        dest="raw_trajs",
        default=[],
        help=(
            "Raw trajectory files as LABEL:GLOB. Repeat for each label. "
            "Quote the argument so the shell does not expand '*'."
        ),
    )
    parser.add_argument(
        "--target-horizons",
        type=str,
        default="50",
        help="Comma-separated prediction horizons to probe, e.g. 1,10,25,50.",
    )
    parser.add_argument(
        "--csv-horizon",
        type=int,
        default=None,
        help=(
            "Horizon used when the prediction CSV was generated. Defaults to "
            "max(--target-horizons). Use 50 for normal multistep CSVs."
        ),
    )
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument(
        "--start-index-offset",
        type=int,
        default=0,
        help="Raw start index offset used by the prediction CSV. Baseline is 0.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["state_hist", "motor_hist", "mode_hist", "delta_u_hist", "actbank"],
        choices=["state_hist", "motor_hist", "mode_hist", "delta_u_hist", "actbank"],
        help="History feature groups to include.",
    )
    parser.add_argument("--actbank-taus-ms", type=str, default="20,50,100,200")
    parser.add_argument("--model", choices=["ridgecv", "ridge", "mlp"], default="ridgecv")
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000",
        help="Comma-separated RidgeCV alpha grid.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["by-time", "random", "by-label"],
        default="by-time",
    )
    parser.add_argument(
        "--test-label",
        action="append",
        dest="test_labels",
        default=[],
        help="For --split-mode by-label, labels held out as test data.",
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--save-design-matrix", action="store_true")
    return parser.parse_args(argv)


def parse_int_list(text, name):
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be > 0.")
    return values


def parse_float_list(text, name):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def validate_args(args):
    if not args.prediction_csvs:
        raise ValueError("At least one --prediction-csv is required.")
    if len(args.prediction_csvs) != len(args.traj_labels):
        raise ValueError("--traj-label count must match --prediction-csv count.")
    if not args.raw_trajs:
        raise ValueError("At least one --raw-traj LABEL:GLOB is required.")
    if args.history_len < 0:
        raise ValueError("--history-len must be >= 0.")
    if args.start_index_offset < 0:
        raise ValueError("--start-index-offset must be >= 0.")
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0.")


def require_columns(df, columns, source):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def parse_raw_traj_specs(raw_trajs):
    specs = {}
    for spec in raw_trajs:
        parts = spec.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"--raw-traj must be LABEL:GLOB, got {spec!r}")
        label, pattern = parts
        files = sorted(Path(path).resolve() for path in glob.glob(os.path.expanduser(pattern)))
        if not files:
            raise FileNotFoundError(f"No raw files matched {pattern!r} for label {label!r}")
        specs[label] = files
    return specs


def motor_to_phys_np(motors):
    motors = np.asarray(motors, dtype=float)
    omega2 = motors ** 2
    thrust = KT * omega2.sum(axis=1)
    tau_x = KT * ARM_LENGTH * (
        (omega2[:, 2] + omega2[:, 3]) - (omega2[:, 0] + omega2[:, 1])
    )
    tau_y = KT * ARM_LENGTH * (
        (omega2[:, 1] + omega2[:, 2]) - (omega2[:, 0] + omega2[:, 3])
    )
    tau_z = KC * ((omega2[:, 0] + omega2[:, 2]) - (omega2[:, 1] + omega2[:, 3]))
    return np.column_stack(
        [thrust / T_MAX, tau_x / MAX_TORQUE[0], tau_y / MAX_TORQUE[1], tau_z / MAX_TORQUE[2]]
    )


def raw_state_matrix(df):
    require_columns(df, RAW_STATE_COLUMNS, "raw trajectory CSV")
    quat = df[["qx", "qy", "qz", "qw"]].to_numpy(float)
    rotvec = R.from_quat(quat).as_rotvec()
    return np.column_stack(
        [
            df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float),
            rotvec,
            df[["wx", "wy", "wz"]].to_numpy(float),
        ]
    )


def flatten_with_names(prefix, values, time_offsets, channel_names):
    values = np.asarray(values, dtype=float)
    names = []
    flat = []
    for t_idx, offset in enumerate(time_offsets):
        for c_idx, channel in enumerate(channel_names):
            names.append(f"{prefix}_t{offset}_{channel}")
            flat.append(values[t_idx, c_idx])
    return flat, names


def actbank_state(motor_hist, taus_ms, dt):
    banks = []
    for tau_ms in taus_ms:
        tau_s = max(float(tau_ms) / 1000.0, 1e-12)
        decay = np.exp(-float(dt) / tau_s)
        bank = motor_hist[0].copy()
        for u_t in motor_hist[1:]:
            bank = decay * bank + (1.0 - decay) * u_t
        banks.append(bank)
    return np.asarray(banks, dtype=float)


def build_feature_vector(raw_df, raw_start, history_len, feature_groups, taus_ms):
    state = raw_state_matrix(raw_df)
    motors = raw_df[MOTOR_COLUMNS].to_numpy(float)
    dt = estimate_dt(raw_df)

    hist_slice = slice(raw_start - history_len, raw_start + 1)
    time_offsets_state = list(range(-history_len, 1))
    features = []
    names = []

    if "state_hist" in feature_groups:
        vals, cols = flatten_with_names(
            "x_hist",
            state[hist_slice],
            time_offsets_state,
            STATE_COLUMNS,
        )
        features.extend(vals)
        names.extend(cols)

    motor_hist = motors[hist_slice]
    if "motor_hist" in feature_groups:
        vals, cols = flatten_with_names(
            "u_hist",
            motor_hist,
            time_offsets_state,
            MOTOR_COLUMNS,
        )
        features.extend(vals)
        names.extend(cols)

    if "mode_hist" in feature_groups:
        vals, cols = flatten_with_names(
            "mode_hist",
            motor_to_phys_np(motor_hist),
            time_offsets_state,
            MODE_COLUMNS,
        )
        features.extend(vals)
        names.extend(cols)

    if "delta_u_hist" in feature_groups:
        delta_u = np.diff(motor_hist, axis=0)
        delta_offsets = list(range(-history_len + 1, 1))
        vals, cols = flatten_with_names(
            "delta_u_hist",
            delta_u,
            delta_offsets,
            MOTOR_COLUMNS,
        )
        features.extend(vals)
        names.extend(cols)

    if "actbank" in feature_groups:
        bank = actbank_state(motor_hist, taus_ms, dt)
        vals, cols = flatten_with_names(
            "actbank",
            bank,
            [f"tau{int(tau)}ms" for tau in taus_ms],
            MOTOR_COLUMNS,
        )
        features.extend(vals)
        names.extend(cols)

    return np.asarray(features, dtype=float), names


def estimate_dt(df):
    if "t" not in df.columns or len(df) < 2:
        return 0.01
    dt_values = np.diff(df["t"].to_numpy(float))
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if dt_values.size == 0:
        return 0.01
    return float(np.median(dt_values))


def build_label_samples(
    label,
    prediction_csv,
    raw_files,
    target_horizons,
    csv_horizon,
    history_len,
    start_index_offset,
    feature_groups,
    taus_ms,
):
    pred_df = require_csv(Path(prediction_csv).expanduser().resolve())
    pred_cols = [
        f"{state}_pred_h{horizon}"
        for horizon in target_horizons
        for state in OMEGA_STATES
    ]
    require_columns(pred_df, pred_cols, str(prediction_csv))

    raw_dfs = []
    expected_counts = []
    for file_path in raw_files:
        raw_df = pd.read_csv(file_path)
        require_columns(raw_df, [*RAW_STATE_COLUMNS, *MOTOR_COLUMNS, "t"], str(file_path))
        expected = len(raw_df) - int(csv_horizon) - int(start_index_offset)
        if expected <= 0:
            raise ValueError(f"{file_path} is too short for csv horizon {csv_horizon}.")
        raw_dfs.append(raw_df)
        expected_counts.append(expected)

    if sum(expected_counts) != len(pred_df):
        detail = {
            str(file_path): count for file_path, count in zip(raw_files, expected_counts)
        }
        raise ValueError(
            "Prediction CSV row count does not match raw files. "
            f"expected {sum(expected_counts)} from raw files, got {len(pred_df)}. "
            f"expected rows by file: {detail}"
        )

    rows = []
    feature_rows = []
    feature_names = None
    pred_cursor = 0

    for file_path, raw_df, expected_count in zip(raw_files, raw_dfs, expected_counts):
        for local_row in range(expected_count):
            pred_row = pred_cursor + local_row
            raw_start = int(start_index_offset + local_row)
            if raw_start < history_len:
                continue

            feature_vector, names = build_feature_vector(
                raw_df,
                raw_start,
                history_len,
                feature_groups,
                taus_ms,
            )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Feature names changed while building samples.")

            if not np.isfinite(feature_vector).all():
                continue

            for horizon in target_horizons:
                target_index = raw_start + horizon
                if target_index >= len(raw_df):
                    continue
                true_omega = raw_df[OMEGA_STATES].iloc[target_index].to_numpy(float)
                pred_omega = pred_df[
                    [f"{state}_pred_h{horizon}" for state in OMEGA_STATES]
                ].iloc[pred_row].to_numpy(float)
                target = true_omega - pred_omega
                if not np.isfinite(target).all():
                    continue
                rows.append(
                    {
                        "traj_label": label,
                        "source_file": str(file_path),
                        "pred_row": pred_row,
                        "raw_start_index": raw_start,
                        "start_time": float(raw_df["t"].iloc[raw_start]),
                        "target_time": float(raw_df["t"].iloc[target_index]),
                        "horizon": horizon,
                        "target_wx": target[0],
                        "target_wy": target[1],
                        "target_wz": target[2],
                    }
                )
                feature_rows.append(feature_vector)
        pred_cursor += expected_count

    if not rows:
        raise ValueError(f"No valid probe samples were built for label {label!r}.")
    return pd.DataFrame(rows), np.vstack(feature_rows), feature_names


def build_samples(args, target_horizons, csv_horizon, taus_ms):
    raw_specs = parse_raw_traj_specs(args.raw_trajs)
    metadata_rows = []
    feature_blocks = []
    feature_names = None
    raw_manifest = {}

    for label, prediction_csv in zip(args.traj_labels, args.prediction_csvs):
        if label not in raw_specs:
            raise ValueError(f"No --raw-traj spec was provided for label {label!r}.")
        raw_files = raw_specs[label]
        raw_manifest[label] = [str(path) for path in raw_files]
        label_df, features, names = build_label_samples(
            label,
            prediction_csv,
            raw_files,
            target_horizons,
            csv_horizon,
            args.history_len,
            args.start_index_offset,
            args.feature_groups,
            taus_ms,
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across trajectory labels.")
        metadata_rows.append(label_df)
        feature_blocks.append(features)

    metadata = pd.concat(metadata_rows, ignore_index=True)
    features = np.vstack(feature_blocks)
    finite = np.isfinite(features).all(axis=1)
    target_cols = ["target_wx", "target_wy", "target_wz"]
    finite &= np.isfinite(metadata[target_cols].to_numpy(float)).all(axis=1)
    metadata = metadata.loc[finite].reset_index(drop=True)
    features = features[finite]

    if args.max_rows is not None and len(metadata) > args.max_rows:
        sampled = metadata.sample(args.max_rows, random_state=args.random_seed).sort_index()
        index = sampled.index.to_numpy()
        metadata = metadata.iloc[index].reset_index(drop=True)
        features = features[index]

    manifest = {
        "prediction_csvs": {
            label: str(Path(path).expanduser().resolve())
            for label, path in zip(args.traj_labels, args.prediction_csvs)
        },
        "raw_files": raw_manifest,
        "target_horizons": target_horizons,
        "csv_horizon": csv_horizon,
        "history_len": args.history_len,
        "start_index_offset": args.start_index_offset,
        "feature_groups": args.feature_groups,
        "row_convention": (
            "Prediction row r starts at raw index r + start_index_offset within each "
            "raw file; pred_h predicts raw index start + h; plain state columns in "
            "prediction CSV are true h=1 and are not used for target construction."
        ),
    }
    return metadata, features, feature_names, manifest


def make_splits(metadata, args):
    rng = np.random.default_rng(args.random_seed)
    train_mask = np.zeros(len(metadata), dtype=bool)
    test_mask = np.zeros(len(metadata), dtype=bool)

    if args.split_mode == "by-label":
        if not args.test_labels:
            raise ValueError("--split-mode by-label requires at least one --test-label.")
        test_labels = set(args.test_labels)
        labels = metadata["traj_label"].to_numpy()
        test_mask = np.isin(labels, list(test_labels))
        train_mask = ~test_mask
    elif args.split_mode == "random":
        groups = metadata[["traj_label", "source_file", "raw_start_index"]].drop_duplicates()
        group_indices = np.arange(len(groups))
        rng.shuffle(group_indices)
        n_train = max(1, int(round(len(group_indices) * args.train_frac)))
        n_train = min(n_train, len(group_indices) - 1)
        train_groups = set(map(tuple, groups.iloc[group_indices[:n_train]].to_numpy()))
        keys = list(map(tuple, metadata[["traj_label", "source_file", "raw_start_index"]].to_numpy()))
        train_mask = np.asarray([key in train_groups for key in keys], dtype=bool)
        test_mask = ~train_mask
    elif args.split_mode == "by-time":
        for _, group in metadata.groupby(["traj_label", "source_file"], sort=False):
            starts = np.sort(group["raw_start_index"].unique())
            n_train = max(1, int(round(len(starts) * args.train_frac)))
            n_train = min(n_train, len(starts) - 1) if len(starts) > 1 else len(starts)
            train_starts = set(starts[:n_train])
            group_train = group["raw_start_index"].isin(train_starts).to_numpy()
            train_mask[group.index.to_numpy()[group_train]] = True
            test_mask[group.index.to_numpy()[~group_train]] = True
    else:
        raise ValueError(f"Unknown split mode {args.split_mode!r}.")

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Split produced train={train_mask.sum()} and test={test_mask.sum()} samples."
        )
    return train_mask, test_mask


def build_model(args):
    if args.model == "ridgecv":
        alphas = parse_float_list(args.alpha_grid, "--alpha-grid")
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if args.model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if args.model == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=args.random_seed,
            ),
        )
    raise ValueError(f"Unknown model {args.model!r}.")


def evaluate_predictions(metadata, y_true, y_pred, split_name, model_name):
    residual_after = y_true - y_pred
    rows = []
    for horizon in sorted(metadata["horizon"].unique()):
        mask = metadata["horizon"].to_numpy() == horizon
        y_h = y_true[mask]
        pred_h = y_pred[mask]
        after_h = residual_after[mask]
        row = {
            "horizon": int(horizon),
            "split": split_name,
            "model": model_name,
            "n_samples": int(mask.sum()),
            "rmse_norm": float(np.sqrt(np.mean(np.sum(after_h ** 2, axis=1)))),
            "zero_correction_rmse_norm": float(np.sqrt(np.mean(np.sum(y_h ** 2, axis=1)))),
            "mae_norm": float(np.mean(np.linalg.norm(after_h, axis=1))),
            "target_norm_mean": float(np.mean(np.linalg.norm(y_h, axis=1))),
        }
        for idx, channel in enumerate(OMEGA_STATES):
            row[f"rmse_{channel}"] = float(
                mean_squared_error(y_h[:, idx], pred_h[:, idx]) ** 0.5
            )
            row[f"mae_{channel}"] = float(mean_absolute_error(y_h[:, idx], pred_h[:, idx]))
            row[f"r2_{channel}"] = float(r2_score(y_h[:, idx], pred_h[:, idx]))
            row[f"zero_correction_rmse_{channel}"] = float(
                np.sqrt(np.mean(y_h[:, idx] ** 2))
            )
        rows.append(row)
    return rows


def coefficient_table(model, feature_names):
    if not hasattr(model, "named_steps"):
        return pd.DataFrame()
    reg = model.named_steps.get("ridgecv") or model.named_steps.get("ridge")
    scaler = model.named_steps.get("standardscaler")
    if reg is None or not hasattr(reg, "coef_"):
        return pd.DataFrame()
    coef = np.asarray(reg.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef[None, :]
    if scaler is not None and hasattr(scaler, "scale_"):
        coef = coef / scaler.scale_[None, :]
    table = pd.DataFrame({"feature": feature_names})
    for idx, channel in enumerate(OMEGA_STATES):
        table[f"coef_{channel}"] = coef[idx]
    table["coef_norm"] = np.linalg.norm(coef, axis=0)
    return table.sort_values("coef_norm", ascending=False)


def plot_target_vs_probe(pred_df, horizon, out_dir):
    setup_matplotlib()
    subset = pred_df[pred_df["horizon"] == horizon].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["traj_label", "source_file", "raw_start_index"])
    x = np.arange(len(subset))

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=180)
    for idx, channel in enumerate(OMEGA_STATES):
        ax = axs[idx]
        ax.plot(x, subset[f"target_{channel}"], color="0.35", linewidth=1.0, label="target")
        ax.plot(
            x,
            subset[f"probe_pred_{channel}"],
            color="tab:blue",
            linewidth=1.0,
            label="probe",
        )
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_ylabel(channel)
        ax.grid(True, alpha=0.3)
    axs[0].legend()
    axs[-1].set_xlabel("Held-out sample order")
    fig.suptitle(f"Oracle residual target vs probe prediction, h={horizon}")
    fig.tight_layout()

    path = out_dir / f"target_vs_probe_timeseries_h{horizon}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_norm_hist(pred_df, horizon, out_dir):
    setup_matplotlib()
    subset = pred_df[pred_df["horizon"] == horizon].copy()
    if subset.empty:
        return None
    target = subset[[f"target_{channel}" for channel in OMEGA_STATES]].to_numpy(float)
    after = subset[[f"resid_after_probe_{channel}" for channel in OMEGA_STATES]].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=180)
    ax.hist(np.linalg.norm(target, axis=1), bins=50, alpha=0.45, label="before probe")
    ax.hist(np.linalg.norm(after, axis=1), bins=50, alpha=0.45, label="after probe")
    ax.set_xlabel("Omega residual norm [rad/s]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / f"target_norm_hist_h{horizon}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rmse_by_horizon(summary_df, out_dir):
    setup_matplotlib()
    test_df = summary_df[summary_df["split"] == "test"].sort_values("horizon")
    if test_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=180)
    ax.plot(
        test_df["horizon"],
        test_df["zero_correction_rmse_norm"],
        "k--",
        linewidth=1.2,
        label="zero correction",
    )
    ax.plot(
        test_df["horizon"],
        test_df["rmse_norm"],
        color="tab:blue",
        linewidth=1.4,
        label="probe",
    )
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Omega residual RMSE norm [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "rmse_by_horizon.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    target_horizons = parse_int_list(args.target_horizons, "--target-horizons")
    csv_horizon = args.csv_horizon if args.csv_horizon is not None else max(target_horizons)
    if csv_horizon < max(target_horizons):
        raise ValueError("--csv-horizon must be >= max(--target-horizons).")
    taus_ms = parse_float_list(args.actbank_taus_ms, "--actbank-taus-ms")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata, features, feature_names, manifest = build_samples(
        args,
        target_horizons,
        csv_horizon,
        taus_ms,
    )
    y = metadata[["target_wx", "target_wy", "target_wz"]].to_numpy(float)
    train_mask, test_mask = make_splits(metadata, args)

    model = build_model(args)
    model.fit(features[train_mask], y[train_mask])
    pred_all = model.predict(features)

    summary_rows = []
    summary_rows.extend(
        evaluate_predictions(
            metadata.loc[train_mask],
            y[train_mask],
            pred_all[train_mask],
            "train",
            args.model,
        )
    )
    summary_rows.extend(
        evaluate_predictions(
            metadata.loc[test_mask],
            y[test_mask],
            pred_all[test_mask],
            "test",
            args.model,
        )
    )
    summary_df = pd.DataFrame(summary_rows)

    pred_df = metadata.copy()
    for idx, channel in enumerate(OMEGA_STATES):
        pred_df[f"probe_pred_{channel}"] = pred_all[:, idx]
        pred_df[f"resid_after_probe_{channel}"] = y[:, idx] - pred_all[:, idx]
    pred_df["split"] = np.where(train_mask, "train", "test")

    summary_df.to_csv(out_dir / "probe_summary.csv", index=False)
    pred_df.to_csv(out_dir / f"probe_predictions_h{max(target_horizons)}.csv", index=False)
    coef_df = coefficient_table(model, feature_names)
    if not coef_df.empty:
        coef_df.to_csv(out_dir / f"feature_importance_h{max(target_horizons)}.csv", index=False)
    if args.save_design_matrix:
        np.savez_compressed(
            out_dir / "probe_design_matrix.npz",
            X=features,
            y=y,
            feature_names=np.asarray(feature_names, dtype=object),
        )

    manifest.update(
        {
            "model": args.model,
            "split_mode": args.split_mode,
            "test_labels": args.test_labels,
            "train_frac": args.train_frac,
            "n_samples": int(len(metadata)),
            "n_features": int(features.shape[1]),
        }
    )
    with open(out_dir / "alignment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plot_paths = [plot_rmse_by_horizon(summary_df, out_dir)]
    for horizon in target_horizons:
        test_pred_df = pred_df[pred_df["split"] == "test"]
        plot_paths.append(plot_target_vs_probe(test_pred_df, horizon, out_dir))
        plot_paths.append(plot_target_norm_hist(test_pred_df, horizon, out_dir))

    print(f"Saved summary to: {out_dir / 'probe_summary.csv'}")
    print(f"Saved predictions to: {out_dir / f'probe_predictions_h{max(target_horizons)}.csv'}")
    if not coef_df.empty:
        print(f"Saved feature importance to: {out_dir / f'feature_importance_h{max(target_horizons)}.csv'}")
    print(f"Saved manifest to: {out_dir / 'alignment_manifest.json'}")
    for path in plot_paths:
        if path is not None:
            print(f"Saved plot to: {path}")


if __name__ == "__main__":
    main()
