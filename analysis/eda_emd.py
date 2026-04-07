"""
无人机飞行数据 — 时间序列可视化 + 经验模态分解（EMD）+ 周期性频谱分析
==========================================================================
用法:
    python analysis/eda_emd.py                           # 默认使用 chirp_20251017_run1.csv
    python analysis/eda_emd.py --csv data/train/random_20251017_run1.csv
    python analysis/eda_emd.py --csv data/train/square_20251017_run2.csv --save
    python analysis/eda_emd.py --show                    # 弹窗显示图形（适合本地桌面环境）
"""

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyEMD import EMD

# ──────────────────────────────────────────────
#  全局配置
# ──────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "train" / "chirp_20251017_run1.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

FEATURE_GROUPS = {
    "位置 (position)":      ["x", "y", "z"],
    "四元数 (quaternion)":  ["qx", "qy", "qz", "qw"],
    "线速度 (velocity)":    ["vx", "vy", "vz"],
    "角速度 (angular vel)": ["wx", "wy", "wz"],
    "电机转速 (motors)":    ["m1_rads", "m2_rads", "m3_rads", "m4_rads"],
    "机体加速度 (body acc)": ["ax_body", "ay_body", "az_body"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="飞行数据 EDA + EMD 分析")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="CSV 文件路径（默认 chirp_20251017_run1.csv）")
    parser.add_argument("--save", action="store_true",
                        help="将图片保存到 plots/ 目录（默认开启）")
    parser.add_argument("--show", action="store_true",
                        help="plt.show() 弹窗显示图形")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="不保存图片")
    parser.set_defaults(save=True)
    return parser.parse_args()


# ──────────────────────────────────────────────
#  1. 加载数据
# ──────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError("CSV 中未找到时间列 't'")
    df = df.sort_values("t").reset_index(drop=True)
    print(f"[INFO] 已加载 {csv_path}")
    print(f"       行数={len(df)}, 列={list(df.columns)}")
    return df


# ──────────────────────────────────────────────
#  2. 绘制各特征的时间序列
# ──────────────────────────────────────────────
def plot_timeseries(df: pd.DataFrame, save_dir: pathlib.Path | None = None):
    """为 df 中每一个非 t 列绘制独立的时序图。"""
    t = df["t"].values
    feature_cols = [c for c in df.columns if c != "t"]

    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, df[col].values, linewidth=0.6)
        ax.set_title(col, fontsize=13)
        ax.set_xlabel("Time t (s)")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"ts_{col}.png", dpi=150)
        plt.close(fig)

    print(f"[INFO] 时间序列图绘制完成，共 {len(feature_cols)} 张")


# ──────────────────────────────────────────────
#  3. 对每个特征做 EMD 分解并绘图
# ──────────────────────────────────────────────
def run_emd_and_plot(df: pd.DataFrame, save_dir: pathlib.Path | None = None):
    """对每列做 EMD，绘制 IMF 子图，并返回 {col: IMFs} 字典。"""
    t = df["t"].values
    feature_cols = [c for c in df.columns if c != "t"]
    emd = EMD()
    all_imfs = {}

    for col in feature_cols:
        signal = df[col].values.astype(np.float64)
        try:
            imfs = emd.emd(signal, t)
        except Exception as exc:
            print(f"[WARN] EMD 分解失败: {col} -> {exc}")
            continue

        n_imfs = imfs.shape[0]
        all_imfs[col] = imfs
        print(f"  {col}: 产生 {n_imfs} 个 IMF")

        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(12, 2.2 * (n_imfs + 1)),
                                 sharex=True)
        if n_imfs + 1 == 1:
            axes = [axes]

        axes[0].plot(t, signal, linewidth=0.6, color="black")
        axes[0].set_title(f"{col} — Original Signal", fontsize=11)
        axes[0].set_ylabel("amplitude")
        axes[0].grid(True, alpha=0.3)

        for i in range(n_imfs):
            axes[i + 1].plot(t, imfs[i], linewidth=0.5)
            axes[i + 1].set_title(f"{col} — IMF {i + 1}", fontsize=10)
            axes[i + 1].set_ylabel("amplitude")
            axes[i + 1].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time t (s)")
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"emd_{col}.png", dpi=150)
        plt.close(fig)

    print(f"[INFO] EMD 分解与绘图完成，共处理 {len(all_imfs)} 个特征")
    return all_imfs


# ──────────────────────────────────────────────
#  4. 周期性分析（FFT）
# ──────────────────────────────────────────────
def periodicity_analysis(df: pd.DataFrame, all_imfs: dict):
    """
    对每个 IMF 做 FFT，找出主频率及其对应周期，
    并打印简单的高频/低频提示。
    """
    t = df["t"].values
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    N = len(t)

    print("\n" + "=" * 72)
    print("  周期性分析（FFT 主频率 & 周期）")
    print("=" * 72)
    print(f"  采样间隔 dt ≈ {dt:.6f} s,  采样率 fs ≈ {fs:.2f} Hz,  样本数 N = {N}")
    print("=" * 72)

    freqs = np.fft.rfftfreq(N, d=dt)

    for col, imfs in all_imfs.items():
        print(f"\n┌─ {col}")
        for i, imf in enumerate(imfs):
            spectrum = np.abs(np.fft.rfft(imf))
            spectrum[0] = 0  # 忽略直流分量

            peak_idx = np.argmax(spectrum)
            peak_freq = freqs[peak_idx]
            peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf

            energy = np.sum(spectrum ** 2)
            high_band = np.sum(spectrum[freqs > fs / 4] ** 2)
            ratio = high_band / energy if energy > 0 else 0

            if peak_freq > fs / 4:
                tag = "高频"
            elif peak_freq > fs / 10:
                tag = "中频"
            else:
                tag = "低频"

            print(f"│  IMF {i + 1:>2d}:  主频 = {peak_freq:8.3f} Hz, "
                  f"周期 = {peak_period:10.4f} s,  "
                  f"高频能量占比 = {ratio:.1%},  [{tag}]")
        print("└" + "─" * 60)


# ──────────────────────────────────────────────
#  主入口
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # 非交互式后端（服务器友好）
    if not args.show:
        matplotlib.use("Agg")

    # 1. 加载数据
    df = load_data(args.csv)

    # 准备输出目录
    save_dir = None
    if args.save:
        csv_stem = pathlib.Path(args.csv).stem
        save_dir = PLOTS_DIR / csv_stem
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 图片将保存到 {save_dir}/")

    # 2. 绘制时间序列
    print("\n>>> 步骤 2: 绘制各特征时间序列 ...")
    plot_timeseries(df, save_dir=save_dir)

    # 3. EMD 分解
    print("\n>>> 步骤 3: 经验模态分解（EMD）...")
    all_imfs = run_emd_and_plot(df, save_dir=save_dir)

    # 4. 周期性分析
    print("\n>>> 步骤 4: 周期性分析（FFT）...")
    periodicity_analysis(df, all_imfs)

    # 完成
    if args.show:
        plt.show()

    print("\n[DONE] 分析完成。")
    if save_dir is not None:
        print(f"       图片已保存到: {save_dir}/")


if __name__ == "__main__":
    main()
