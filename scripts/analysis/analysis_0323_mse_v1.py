#!/usr/bin/env python3
"""
2026-03-23 三方对比分析：HPIPM vs PMP vs PMP+MLP(MSE v1, 递归3步)
数据来源：320_4 ~ 320_6（含 err_pos_y, err_yaw 列）

用法：
    cd /home/markwcl/MPC_ML
    python3 analysis_fixed_mse_v1.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── 数据文件路径 ──────────────────────────────────────────────────────────────
HPIPM_CSVS = [
    "/home/markwcl/Bipedal_MPC/eval/509_1/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/509_2/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/509_3/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/509_4/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/509_5/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/510_1/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/510_2/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/510_3/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/510_4/hpipm_eval_result.csv",
    "/home/markwcl/Bipedal_MPC/eval/510_5/hpipm_eval_result.csv"
]
PMP_CSVS = [
    "/home/markwcl/My_MPC/eval/509_1/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/509_2/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/509_3/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/509_4/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_1/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_2/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_3/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_4/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_5/pmp_eval_result.csv",
    "/home/markwcl/My_MPC/eval/510_6/pmp_eval_result.csv"
]
PMPML_CSVS = [
    "/home/markwcl/MPC_ML/eval/510_1/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_2/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_3/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_4/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_5/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_6/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_7/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_8/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_9/pmp_ml_eval_result.csv",
    "/home/markwcl/MPC_ML/eval/510_10/pmp_ml_eval_result.csv"
]

OUT_DIR = Path("/home/markwcl/MPC_ML/img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["HPIPM", "PMP", "PMP+MLP"]
ALL_CSVS = [HPIPM_CSVS, PMP_CSVS, PMPML_CSVS]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]  # blue / orange / green

plt.rcParams.update({
    'font.size': 10,
    'axes.grid': False,
    'figure.dpi': 150,
})

# ─── 加载数据 ──────────────────────────────────────────────────────────────────
def load_and_merge(csv_list):
    dfs = []
    for p in csv_list:
        df = pd.read_csv(p)
        df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
        dfs.append(df)
    return dfs

all_data = {}  # label -> list of DataFrames (per run)
all_merged = {}  # label -> single merged DataFrame
for label, csvs in zip(LABELS, ALL_CSVS):
    runs = load_and_merge(csvs)
    all_data[label] = runs
    all_merged[label] = pd.concat(runs, ignore_index=True)
    print(f"[OK] {label}: {len(runs)} runs, {sum(len(r) for r in runs)} total samples")

# ─── 图1：pos_y 和 yaw 时序对比（核心图）────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Lateral Tracking: pos_y Error & yaw Error\n(10 runs averaged, MSE-trained MLP, recursive 3 steps)",
             fontsize=13, fontweight="bold")

for ax, col, ylabel, title in zip(
    axes,
    ["err_pos_y", "err_yaw"],
    ["pos_y tracking error", "yaw tracking error"],
    ["(a) Lateral position error (pos_y)", "(b) Heading error (yaw)"]):

    for label, runs, color in zip(LABELS, [all_data[l] for l in LABELS], COLORS):
        # plot each run faintly
        for run_df in runs:
            ax.plot(run_df["t"].values, run_df[col].values,
                    color=color, alpha=0.08, linewidth=0.4)
        # smoothed average across runs
        min_len = min(len(r) for r in runs)
        avg_vals = np.mean([r[col].values[:min_len] for r in runs], axis=0)
        t_vals = runs[0]["t"].values[:min_len]
        w = min(50, len(avg_vals) // 4)
        sm = np.convolve(avg_vals, np.ones(w) / w, mode="same")
        mean_val = np.mean(avg_vals)
        ax.plot(t_vals, sm, color=color, linewidth=2.0,
                label=f"{label} (mean={mean_val:.4f})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
out1 = OUT_DIR / "fixed_lateral_tracking.png"
plt.savefig(out1, dpi=200, bbox_inches="tight")
print(f"[saved] {out1}")
plt.close()

# ─── 图2：pos_y 全分位数 bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("pos_y Tracking Error: Full Percentile Comparison\n(lower is better = straighter walking)",
             fontsize=12, fontweight="bold")

stat_names = ["mean", "median", "p90", "p95", "p99"]
stat_fns = {
    "mean": np.mean, "median": np.median,
    "p90": lambda x: np.percentile(x, 90),
    "p95": lambda x: np.percentile(x, 95),
    "p99": lambda x: np.percentile(x, 99),
}

x_pos = np.arange(len(stat_names))
bar_width = 0.25

for i, (label, color) in enumerate(zip(LABELS, COLORS)):
    vals = [stat_fns[s](all_merged[label]["err_pos_y"].values) for s in stat_names]
    bars = ax.bar(x_pos + i * bar_width, vals, bar_width,
                  label=label, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(stat_names)
ax.set_ylabel("err_pos_y")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.2, linestyle="--")
ax.set_ylim(bottom=0)

plt.tight_layout()
out2 = OUT_DIR / "fixed_posy_percentiles.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
print(f"[saved] {out2}")
plt.close()

# ─── 图3：综合 bar chart（给导师看的总览图）──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Three-way Comparison: HPIPM vs PMP vs PMP+MLP\n"
             "(10 runs × 20s each)",
             fontsize=12, fontweight="bold", y=1.02)

bar_metrics = [
    ("avg_tracking_error", "Avg Tracking Error", "mean", ".4f", True),
    ("err_pos_y",          "Lateral Error (pos_y)", "mean", ".4f", True),
    ("err_yaw",            "Heading Error (yaw)", "mean", ".4f", True),
    ("avg_delta_u",        "Avg Control Variation (N)", "p99", ".2f", True),
    None,  # panel (e): startup pass rate, fixed-100-trial check (n=100, not 10-run stat)
    ("solve_time_ms",      "Solve Time (ms)", "mean", ".2f", True),
]

# Init stability data: fixed-100-trial startup check (max e^infty < 5 in first 1s)
# Wilson 95% CI computed from n=100 binomial counts (48/100, 59/100, 97/100).
startup_rates    = np.array([48.0, 59.0, 97.0])
startup_counts   = ["48/100", "59/100", "97/100"]
startup_ci_low   = np.array([38.5, 49.2, 91.5])
startup_ci_high  = np.array([57.7, 68.1, 99.0])
startup_yerr     = np.vstack([startup_rates - startup_ci_low,
                              startup_ci_high - startup_rates])

for panel_idx, (ax, metric) in enumerate(zip(axes.flat, bar_metrics)):
    panel_letter = chr(ord("a") + panel_idx)
    ax.text(-0.10, 1.04, f"({panel_letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    if metric is None:
        bars = ax.bar(LABELS, startup_rates, color=COLORS, alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=startup_yerr, capsize=4,
                      error_kw=dict(linewidth=1.0, capthick=1.0))
        for bar, label in zip(bars, startup_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2.0,
                    label, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        ax.text(2, startup_rates[2] * 0.5,
                "+49 pp\nvs HPIPM",
                ha="center", va="center", fontsize=8,
                color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8))
        ax.set_ylabel("Startup pass rate (%)", fontsize=9)
        ax.set_title("Startup Pass Rate (100 trials)", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="x", labelsize=8.5)
        continue

    col, ylabel, stat, fmt, lower_better = metric
    fn = stat_fns.get(stat, np.mean)
    means = [fn(all_merged[l][col].values) for l in LABELS]

    bars = ax.bar(LABELS, means, color=COLORS, alpha=0.85,
                  edgecolor="black", linewidth=0.8)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.01,
                f"{m:{fmt}}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # annotate PMP+MLP vs HPIPM
    ref, ml = means[0], means[2]
    pct = (ml - ref) / ref * 100
    sign = "+" if pct > 0 else ""
    improved = (lower_better and pct < 0) or (not lower_better and pct > 0)
    color_ann = "red" if improved else "gray"
    ax.text(2, means[2] * 0.5,
            f"{sign}{pct:.1f}%\nvs HPIPM",
            ha="center", va="center", fontsize=8,
            color=color_ann, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_ann, alpha=0.8))

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel} ({stat})", fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelsize=8.5)

plt.tight_layout()
out3 = OUT_DIR / "fixed_summary_6panel.png"
plt.savefig(out3, dpi=200, bbox_inches="tight")
print(f"[saved] {out3}")
plt.close()

# ─── 图4：箱线图（pos_y 和 yaw 对比）─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribution of Lateral & Heading Errors (box plots)", fontsize=13, fontweight="bold")

for ax, col, ylabel in zip(axes,
    ["err_pos_y", "err_yaw"],
    ["pos_y error", "yaw error"]):
    data = [all_merged[l][col].values for l in LABELS]
    bp = ax.boxplot(data, labels=LABELS, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    showfliers=False)  # hide outliers for clarity
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    # annotate medians
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med, f"  {med:.4f}", va="center", fontsize=8, color="black")

plt.tight_layout()
out4 = OUT_DIR / "fixed_lateral_boxplot.png"
plt.savefig(out4, dpi=200, bbox_inches="tight")
print(f"[saved] {out4}")
plt.close()

# ─── 图5：avg_tracking 和 max_tracking 时序 ─────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Tracking Error Over Time\n(10 runs averaged)", fontsize=13, fontweight="bold")

for ax, col, ylabel, title in zip(
    axes,
    ["avg_tracking_error", "max_tracking_error"],
    ["avg tracking error (L1 mean)", "max tracking error (inf-norm)"],
    ["(a) Average tracking error", "(b) Maximum tracking error"]):

    for label, runs, color in zip(LABELS, [all_data[l] for l in LABELS], COLORS):
        for run_df in runs:
            ax.plot(run_df["t"].values, run_df[col].values,
                    color=color, alpha=0.06, linewidth=0.3)
        min_len = min(len(r) for r in runs)
        avg_vals = np.mean([r[col].values[:min_len] for r in runs], axis=0)
        t_vals = runs[0]["t"].values[:min_len]
        w = min(50, len(avg_vals) // 4)
        sm = np.convolve(avg_vals, np.ones(w) / w, mode="same")
        mean_val = np.mean(avg_vals)
        ax.plot(t_vals, sm, color=color, linewidth=2.0,
                label=f"{label} (mean={mean_val:.4f})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
out5 = OUT_DIR / "fixed_tracking_timeseries.png"
plt.savefig(out5, dpi=200, bbox_inches="tight")
print(f"[saved] {out5}")
plt.close()

print(f"\n[done] All figures saved to {OUT_DIR}")
