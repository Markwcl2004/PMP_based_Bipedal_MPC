#!/usr/bin/env python3
"""
三方对比分析：HPIPM vs 纯PMP vs PMP+ML(方案C)
读取各项目的 *_eval_result.csv，输出求解时间、跟踪误差、约束违反统计与图表。

用法：
    cd /home/markwcl/MPC_ML
    python3 analysis_threeway.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── 数据文件路径 ──────────────────────────────────────────────────────────────
CSV_HPIPM  = "/home/markwcl/Bipedal_MPC/hpipm_eval_result.csv"
CSV_PMP    = "/home/markwcl/My_MPC/pmp_eval_result.csv"
CSV_PMPML  = "/home/markwcl/MPC_ML/pmp_ml_eval_result.csv"
OUT_DIR    = Path("/home/markwcl/MPC_ML/img")

LABELS     = ["HPIPM",  "PMP",    "PMP+ML(C')"]
CSV_PATHS  = [CSV_HPIPM, CSV_PMP, CSV_PMPML]
COLORS     = ["#2196F3", "#FF9800", "#4CAF50"]   # blue / orange / green

# ─── 加载数据 ──────────────────────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    # 相对时间（从0开始）
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df

dfs = []
for label, path in zip(LABELS, CSV_PATHS):
    try:
        df = load_csv(path)
        dfs.append(df)
        print(f"[OK] {label}: {len(df)} samples  "
              f"(t0={df['timestamp'].iloc[0]:.0f}, span={df['t'].iloc[-1]:.1f}s)")
    except FileNotFoundError:
        print(f"[MISS] {label}: {path} 不存在，请先跑对应 launch")
        dfs.append(None)

# ─── 统计摘要 ──────────────────────────────────────────────────────────────────
def stats(series, fmt=".3f"):
    return (f"mean={series.mean():{fmt}}  "
            f"median={series.median():{fmt}}  "
            f"std={series.std():{fmt}}  "
            f"min={series.min():{fmt}}  "
            f"max={series.max():{fmt}}")

print("\n" + "="*70)
print("  求解时间  solve_time_ms")
print("="*70)
for label, df in zip(LABELS, dfs):
    if df is not None:
        print(f"  {label:<14} {stats(df['solve_time_ms'])}")

print("\n" + "="*70)
print("  平均跟踪误差  avg_tracking_error")
print("="*70)
for label, df in zip(LABELS, dfs):
    if df is not None:
        print(f"  {label:<14} {stats(df['avg_tracking_error'])}")

print("\n" + "="*70)
print("  最大约束违反  max_constraint_violation")
print("="*70)
for label, df in zip(LABELS, dfs):
    if df is not None:
        print(f"  {label:<14} {stats(df['max_constraint_violation'], fmt='.2f')}")

# ─── 对比表 ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  {'指标':<30} {'HPIPM':>12} {'PMP':>12} {'PMP+ML(C)':>12}")
print("-"*70)

valid = [i for i, df in enumerate(dfs) if df is not None]
metrics = [
    ("solve_time_ms (mean)",         "solve_time_ms",             ".3f"),
    ("solve_time_ms (median)",       "solve_time_ms",             ".3f"),
    ("solve_time_ms (std)",          "solve_time_ms",             ".3f"),
    ("avg_tracking_error (mean)",    "avg_tracking_error",        ".4f"),
    ("max_tracking_error (mean)",    "max_tracking_error",        ".4f"),
    ("max_constraint_viol (mean)",   "max_constraint_violation",  ".2f"),
]
agg_fns = {"mean": np.mean, "median": np.median, "std": np.std}

for disp, col, fmt in metrics:
    fn_name = disp.split("(")[1].rstrip(")")
    fn = agg_fns.get(fn_name, np.mean)
    row = [f"{fn(dfs[i][col]):{fmt}}" if dfs[i] is not None else "N/A"
           for i in range(3)]
    print(f"  {disp:<30} {row[0]:>12} {row[1]:>12} {row[2]:>12}")

# 求解时间相对 HPIPM 的提升
if dfs[0] is not None:
    ref = dfs[0]["solve_time_ms"].mean()
    print(f"\n  求解时间 vs HPIPM（mean）:")
    for label, df in zip(LABELS, dfs):
        if df is not None:
            ratio = df["solve_time_ms"].mean() / ref
            arrow = "▼ 慢" if ratio > 1.05 else ("▲ 快" if ratio < 0.95 else "≈ 持平")
            print(f"    {label:<14} {ratio:.2f}x  {arrow}")

print("="*70)

# ─── 图1：求解时间时序 + 分布 ──────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Solve Time Comparison (same-session data)", fontsize=13, fontweight="bold")

ax_ts = axes[0]
ax_cdf = axes[1]

for label, df, color in zip(LABELS, dfs, COLORS):
    if df is None:
        continue
    t = df["t"].values
    st = df["solve_time_ms"].values
    ax_ts.plot(t, st, color=color, alpha=0.25, linewidth=0.6)
    # 0.5s 移动均值（约50个样本@100Hz）
    w = min(50, len(st)//4)
    sm = np.convolve(st, np.ones(w)/w, mode="same")
    ax_ts.plot(t, sm, color=color, linewidth=1.8, label=f"{label} (mean={st.mean():.2f}ms)")

ax_ts.set_xlabel("Time (s)")
ax_ts.set_ylabel("Solve time (ms)")
ax_ts.set_title("(a) Solve time over time")
ax_ts.legend(fontsize=9)
ax_ts.grid(True, alpha=0.3)

for label, df, color in zip(LABELS, dfs, COLORS):
    if df is None:
        continue
    st = np.sort(df["solve_time_ms"].values)
    cdf = np.arange(1, len(st)+1) / len(st)
    ax_cdf.plot(st, cdf, color=color, linewidth=1.8, label=label)

ax_cdf.set_xlabel("Solve time (ms)")
ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("(b) Solve time CDF")
ax_cdf.legend(fontsize=9)
ax_cdf.grid(True, alpha=0.3)
ax_cdf.set_xlim(left=0)

plt.tight_layout()
out1 = OUT_DIR / "threeway_solve_time.png"
plt.savefig(out1, dpi=180, bbox_inches="tight")
print(f"\n[保存] {out1}")
plt.close()

# ─── 图2：跟踪误差 & 约束违反 ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("Tracking Error & Constraint Violation", fontsize=13, fontweight="bold")

for ax, col, ylabel, title_label in zip(
        axes,
        ["avg_tracking_error", "max_constraint_violation"],
        ["avg tracking error", "max constraint violation (N)"],
        ["(a) Average tracking error", "(b) Max constraint violation"]):
    for label, df, color in zip(LABELS, dfs, COLORS):
        if df is None:
            continue
        t = df["t"].values
        v = df[col].values
        ax.plot(t, v, color=color, alpha=0.2, linewidth=0.5)
        w = min(50, len(v)//4)
        sm = np.convolve(v, np.ones(w)/w, mode="same")
        ax.plot(t, sm, color=color, linewidth=1.8,
                label=f"{label} (mean={v.mean():.3f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = OUT_DIR / "threeway_tracking.png"
plt.savefig(out2, dpi=180, bbox_inches="tight")
print(f"[保存] {out2}")
plt.close()

# ─── 图3：箱线图汇总 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Distribution Summary (box plots)", fontsize=13, fontweight="bold")

box_metrics = [
    ("solve_time_ms",          "Solve time (ms)"),
    ("avg_tracking_error",     "Avg tracking error"),
    ("max_constraint_violation","Max constraint viol. (N)"),
]

for ax, (col, ylabel) in zip(axes, box_metrics):
    data = [df[col].values for df in dfs if df is not None]
    lbls = [l for l, df in zip(LABELS, dfs) if df is not None]
    bp = ax.boxplot(data, labels=lbls, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], [c for c, df in zip(COLORS, dfs) if df is not None]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out3 = OUT_DIR / "threeway_boxplot.png"
plt.savefig(out3, dpi=180, bbox_inches="tight")
print(f"[保存] {out3}")
plt.close()

print("\n[完成] 三方对比分析完毕。")

# ─── 图4：汇总 bar chart（给导师看的单张总览图）─────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Three-way MPC Comparison: HPIPM vs PMP vs PMP+ML(C')\n(same-session evaluation, bipedal robot walking)",
             fontsize=12, fontweight="bold", y=1.02)

bar_metrics = [
    ("solve_time_ms",           "Solve Time (ms)",           ".2f", False),
    ("avg_tracking_error",      "Avg Tracking Error",        ".4f", True),
    ("max_tracking_error",      "Max Tracking Error",        ".4f", True),
    ("max_constraint_violation","Max Constraint Viol. (N)",  ".1f", True),
]

valid_labels = [l for l, df in zip(LABELS, dfs) if df is not None]
valid_colors = [c for c, df in zip(COLORS, dfs) if df is not None]
valid_dfs    = [df for df in dfs if df is not None]

for ax, (col, ylabel, fmt, lower_better) in zip(axes, bar_metrics):
    means = [df[col].mean() for df in valid_dfs]
    stds  = [df[col].std()  for df in valid_dfs]

    bars = ax.bar(valid_labels, means, yerr=stds,
                  color=valid_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8,
                  capsize=5, error_kw=dict(linewidth=1.2))

    # 在每个 bar 上标注数值
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + max(means)*0.01,
                f"{mean:{fmt}}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # 标注相对 PMP 的提升（如果 PMP+ML 存在）
    if len(valid_dfs) == 3:
        ref = means[1]   # PMP
        ml  = means[2]   # PMP+ML
        delta_pct = (ml - ref) / ref * 100
        sign = "↑" if delta_pct > 0 else "↓"
        color_ann = "green" if (lower_better and delta_pct < 0) else \
                    ("red"  if lower_better and delta_pct > 0 else "gray")
        ax.text(2, means[2] * 0.5,
                f"{sign}{abs(delta_pct):.1f}%\nvs PMP",
                ha="center", va="center", fontsize=8,
                color=color_ann, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_ann, alpha=0.8))

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", labelsize=8.5)
    note = "lower is better" if lower_better else "lower is faster"
    ax.text(0.98, 0.97, f"({'lower is better' if lower_better else 'lower = faster'})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="gray", style="italic")

plt.tight_layout()
out4 = OUT_DIR / "threeway_summary.png"
plt.savefig(out4, dpi=200, bbox_inches="tight")
print(f"[保存] {out4}")
plt.close()
