#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文 Figure 1：HPIPM vs PMP (无 MLP) 对比
风格与 generate_paper_figures.py / analysis_threeway.py 一致

数据源：
  HPIPM: /home/markwcl/Bipedal_MPC/hpipm_eval_result.csv
  PMP:   /home/markwcl/My_MPC/pmp_eval_result.csv

输出：
  img/image1.png — 2 个子图（垂直）
    (a) Tracking error: avg (solid) + p99 水平线 (dashed)
    (b) Control input variation: avg (solid) + max (dashed)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path):
    df = pd.read_csv(path)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def main():
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = Path(__file__).parent

    # ── 加载数据 ──
    hpipm = load_csv("/home/markwcl/Bipedal_MPC/hpipm_eval_result.csv")
    pmp   = load_csv("/home/markwcl/My_MPC/pmp_eval_result.csv")

    LABELS = ["HPIPM", "PMP"]
    DFS    = [hpipm, pmp]
    COLORS = ["#2196F3", "#FF9800"]

    # ── 统计 ──
    for label, df in zip(LABELS, DFS):
        avg = df["avg_tracking_error"]
        print(f"{label}: avg_mean={avg.mean():.4f}, avg_p99={avg.quantile(0.99):.4f}, "
              f"avg_du={df['avg_delta_u'].mean():.2f}, max_du={df['max_delta_u'].mean():.2f}")

    # ── 绘图 ──
    fig, (ax_trk, ax_u) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("HPIPM vs PMP", fontsize=13, fontweight="bold")

    w = 50  # 移动均值窗口

    # ── (a) Tracking error: avg (solid) + p99 (dashed) ──
    for label, df, color in zip(LABELS, DFS, COLORS):
        t = df["t"].values
        v = df["avg_tracking_error"].values
        p99_val = np.percentile(v, 99)

        # 原始数据浅色
        ax_trk.plot(t, v, color=color, alpha=0.2, linewidth=0.5)
        # 移动均值加粗实线
        ww = min(w, len(v) // 4)
        sm = np.convolve(v, np.ones(ww) / ww, mode="same")
        ax_trk.plot(t, sm, color=color, linewidth=1.8,
                    label=f"{label} avg (mean={v.mean():.3f})")
        # p99 水平虚线
        ax_trk.axhline(y=p99_val, color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                        label=f"{label} p99={p99_val:.3f}")

    ax_trk.set_ylabel("avg tracking error")
    ax_trk.set_title("(a) Average tracking error (solid: mean, dashed: p99)")
    ax_trk.legend(fontsize=9)
    ax_trk.grid(True, alpha=0.3)

    # ── (b) Control input variation: avg (solid) + max (dashed) ──
    for label, df, color in zip(LABELS, DFS, COLORS):
        t = df["t"].values
        v_avg = df["avg_delta_u"].values
        v_max = df["max_delta_u"].values
        ww = min(w, len(v_avg) // 4)

        # avg: 原始浅色 + 移动均值实线
        ax_u.plot(t, v_avg, color=color, alpha=0.15, linewidth=0.5)
        sm_avg = np.convolve(v_avg, np.ones(ww) / ww, mode="same")
        ax_u.plot(t, sm_avg, color=color, linewidth=1.8,
                  label=f"{label} avg (mean={v_avg.mean():.1f}N)")

        # max: 原始浅色 + 移动均值虚线
        ax_u.plot(t, v_max, color=color, alpha=0.1, linewidth=0.4)
        sm_max = np.convolve(v_max, np.ones(ww) / ww, mode="same")
        ax_u.plot(t, sm_max, color=color, linestyle="--", linewidth=1.2, alpha=0.7,
                  label=f"{label} max (mean={v_max.mean():.1f}N)")

    ax_u.set_xlabel("Time (s)")
    ax_u.set_ylabel("Control input variation (N)")
    ax_u.set_title("(b) Control input variation (solid: avg $\\overline{\\Delta u}_k$, dashed: max $\\Delta u_k^\\infty$)")
    ax_u.legend(fontsize=8, ncol=2)
    ax_u.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = root / "img" / "image1.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
