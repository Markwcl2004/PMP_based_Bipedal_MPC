#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026-03-20 valid-run plotting script.
Uses only 320_4 ~ 320_6.
Current eval CSVs do not contain raw x-y trajectory, so the lateral figure uses
err_pos_y / err_yaw as lateral-deviation proxies.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/markwcl/MPC_ML')
OUT_DIR = ROOT / 'img'
RUN_TAGS = ['320_4', '320_5', '320_6']

PROJECTS = {
    'HPIPM': {
        'pattern': '/home/markwcl/Bipedal_MPC/eval/{tag}/hpipm_eval_result.csv',
        'color': '#2196F3',
    },
    'PMP': {
        'pattern': '/home/markwcl/My_MPC/eval/{tag}/pmp_eval_result.csv',
        'color': '#FF9800',
    },
    'PMP+MLP': {
        'pattern': '/home/markwcl/MPC_ML/eval/{tag}/pmp_ml_eval_result.csv',
        'color': '#4CAF50',
    },
}

SUMMARY_METRICS = [
    ('avg_tracking_error', 'Avg tracking error'),
    ('max_tracking_error', 'Max tracking error'),
    ('avg_delta_u', 'Avg delta_u (N)'),
    ('max_delta_u', 'Max delta_u (N)'),
    ('max_constraint_violation', 'Max constraint viol. (N)'),
    ('solve_time_ms', 'Solve time (ms)'),
    ('err_pos_y', 'pos_y abs error'),
    ('err_yaw', 'yaw abs error'),
]


def load_runs():
    nested = {}
    flat = {}
    for method, cfg in PROJECTS.items():
        nested[method] = {}
        parts = []
        for tag in RUN_TAGS:
            path = Path(cfg['pattern'].format(tag=tag))
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path).copy()
            df['run_tag'] = tag
            df['method'] = method
            df['t'] = df['timestamp'] - df['timestamp'].iloc[0]
            nested[method][tag] = df
            parts.append(df)
        flat[method] = pd.concat(parts, ignore_index=True)
    return nested, flat


def summarize(nested_frames):
    per_run_rows = []
    for method, runs in nested_frames.items():
        for run_tag, df in runs.items():
            for metric, _ in SUMMARY_METRICS:
                series = df[metric]
                per_run_rows.append({
                    'method': method,
                    'run_tag': run_tag,
                    'metric': metric,
                    'rows': len(series),
                    'mean': series.mean(),
                    'median': series.median(),
                    'p90': series.quantile(0.90),
                    'p95': series.quantile(0.95),
                    'p99': series.quantile(0.99),
                })

    per_run_df = pd.DataFrame(per_run_rows)
    summary_df = (
        per_run_df
        .groupby(['method', 'metric'], as_index=False)
        .agg({
            'rows': 'mean',
            'mean': 'mean',
            'median': 'mean',
            'p90': 'mean',
            'p95': 'mean',
            'p99': 'mean',
        })
    )
    return summary_df, per_run_df


def summary_row(summary_df, method, metric):
    return summary_df[(summary_df['method'] == method) & (summary_df['metric'] == metric)].iloc[0]


def moving_average(values, window):
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode='same')


def compute_aligned_mean_curve(nested_frames, metric, num_points=500):
    max_ends = [float(df['t'].iloc[-1]) for runs in nested_frames.values() for df in runs.values()]
    t_end = min(max_ends)
    grid = np.linspace(0.0, t_end, num_points)

    curves = {}
    for method, runs in nested_frames.items():
        aligned = []
        for tag in RUN_TAGS:
            df = runs[tag]
            aligned.append(np.interp(grid, df['t'].to_numpy(), df[metric].to_numpy()))
        curves[method] = np.vstack(aligned).mean(axis=0)
    return grid, curves


def format_relative_note(mean_gain=None, p99_gain=None):
    lines = ['PMP+MLP rel. to HPIPM', 'negative = lower metric']
    if mean_gain is not None:
        lines.append(f'mean {mean_gain:+.1f}%')
    if p99_gain is not None:
        lines.append(f'p99 {p99_gain:+.1f}%')
    return '\n'.join(lines)


def make_summary_plot(summary_df):
    import matplotlib.pyplot as plt

    methods = list(PROJECTS.keys())
    colors = [PROJECTS[m]['color'] for m in methods]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.suptitle('Metric Summary', fontsize=13, fontweight='bold')

    x = np.arange(len(methods))
    width = 0.34

    for ax, (metric, title) in zip(axes.flat, SUMMARY_METRICS):
        metric_df = summary_df[summary_df['metric'] == metric].set_index('method').loc[methods]
        means = metric_df['mean'].to_numpy()
        p99s = metric_df['p99'].to_numpy()

        ax.bar(x - width / 2, means, width=width, color=colors, alpha=0.88, label='mean')
        ax.bar(x + width / 2, p99s, width=width, color=colors, alpha=0.35, label='p99')

        for xpos, val in zip(x - width / 2, means):
            ax.text(xpos, val, f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        for xpos, val in zip(x + width / 2, p99s):
            ax.text(xpos, val, f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis='y', alpha=0.25)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color='black', alpha=0.88, label='mean'),
        plt.Rectangle((0, 0), 1, 1, color='black', alpha=0.35, label='p99'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.96))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_raw_and_smoothed(ax, nested_frames, summary_df, metric, title, ylabel, smooth_window, annotate=None):
    methods = list(PROJECTS.keys())
    grid, mean_curves = compute_aligned_mean_curve(nested_frames, metric)

    for method in methods:
        color = PROJECTS[method]['color']
        row = summary_row(summary_df, method, metric)

        for tag in RUN_TAGS:
            df = nested_frames[method][tag]
            ax.plot(df['t'].to_numpy(), df[metric].to_numpy(), color=color, alpha=0.10, linewidth=0.7)

        smooth = moving_average(mean_curves[method], smooth_window)
        ax.plot(grid, smooth, color=color, linewidth=2.0,
                label=f"{method} mean={row['mean']:.4f}")
        ax.axhline(row['p99'], color=color, linestyle='--', linewidth=1.1, alpha=0.75,
                   label=f"{method} p99={row['p99']:.4f}")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc='upper right',
        fontsize=7.2,
        ncol=2,
        frameon=True,
        framealpha=0.92,
        borderpad=0.3,
        handlelength=1.8,
        columnspacing=0.9,
        labelspacing=0.25,
    )
    if annotate is not None:
        ax.text(
            0.02,
            0.97,
            annotate,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8.6,
            fontweight='bold',
            color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.85),
        )


def make_tracking_paper_plot(nested_frames, summary_df):
    import matplotlib.pyplot as plt

    hp_avg = summary_row(summary_df, 'HPIPM', 'avg_tracking_error')
    ml_avg = summary_row(summary_df, 'PMP+MLP', 'avg_tracking_error')
    avg_mean_gain = (ml_avg['mean'] - hp_avg['mean']) / hp_avg['mean'] * 100
    avg_p99_gain = (ml_avg['p99'] - hp_avg['p99']) / hp_avg['p99'] * 100

    hp_max = summary_row(summary_df, 'HPIPM', 'max_tracking_error')
    ml_max = summary_row(summary_df, 'PMP+MLP', 'max_tracking_error')
    max_p99_gain = (ml_max['p99'] - hp_max['p99']) / hp_max['p99'] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True)
    fig.suptitle('Tracking Error', fontsize=13, fontweight='bold')

    plot_raw_and_smoothed(
        ax1, nested_frames, summary_df, 'avg_tracking_error',
        '(a) Mean tracking error', 'avg_tracking_error', smooth_window=17,
        annotate=format_relative_note(mean_gain=avg_mean_gain, p99_gain=avg_p99_gain)
    )
    plot_raw_and_smoothed(
        ax2, nested_frames, summary_df, 'max_tracking_error',
        '(b) Peak tracking error', 'max_tracking_error', smooth_window=17,
        annotate=format_relative_note(p99_gain=max_p99_gain)
    )

    ax2.set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_lateral_proxy_plot(nested_frames, summary_df):
    import matplotlib.pyplot as plt

    hp_pos = summary_row(summary_df, 'HPIPM', 'err_pos_y')
    ml_pos = summary_row(summary_df, 'PMP+MLP', 'err_pos_y')
    pos_mean_gain = (ml_pos['mean'] - hp_pos['mean']) / hp_pos['mean'] * 100
    pos_p99_gain = (ml_pos['p99'] - hp_pos['p99']) / hp_pos['p99'] * 100

    hp_yaw = summary_row(summary_df, 'HPIPM', 'err_yaw')
    ml_yaw = summary_row(summary_df, 'PMP+MLP', 'err_yaw')
    yaw_mean_gain = (ml_yaw['mean'] - hp_yaw['mean']) / hp_yaw['mean'] * 100
    yaw_p99_gain = (ml_yaw['p99'] - hp_yaw['p99']) / hp_yaw['p99'] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True)
    fig.suptitle('Lateral Tracking Error', fontsize=13, fontweight='bold')

    plot_raw_and_smoothed(
        ax1, nested_frames, summary_df, 'err_pos_y',
        '(a) Lateral position error', 'err_pos_y', smooth_window=17,
        annotate=format_relative_note(mean_gain=pos_mean_gain, p99_gain=pos_p99_gain)
    )
    plot_raw_and_smoothed(
        ax2, nested_frames, summary_df, 'err_yaw',
        '(b) Heading error', 'err_yaw', smooth_window=17,
        annotate=format_relative_note(mean_gain=yaw_mean_gain, p99_gain=yaw_p99_gain)
    )

    ax2.set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_straightness_plot(flat_frames, summary_df):
    import matplotlib.pyplot as plt

    methods = list(PROJECTS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle('Distribution of Lateral Metrics', fontsize=13, fontweight='bold')

    for ax, metric, title in [
        (axes[0], 'err_pos_y', '(a) err_pos_y CDF'),
        (axes[1], 'err_yaw', '(b) err_yaw CDF'),
    ]:
        for method in methods:
            df = flat_frames[method]
            values = np.sort(df[metric].to_numpy())
            cdf = np.arange(1, len(values) + 1) / len(values)
            row = summary_row(summary_df, method, metric)
            ax.plot(values, cdf, color=PROJECTS[method]['color'], linewidth=1.8,
                    label=f"{method} mean={row['mean']:.4f}, p99={row['p99']:.4f}")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel('CDF')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main():
    import matplotlib
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nested_frames, flat_frames = load_runs()
    summary_df, per_run_summary_df = summarize(nested_frames)

    summary_path = OUT_DIR / '20260320_mse_v1_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    per_run_summary_path = OUT_DIR / '20260320_mse_v1_per_run_summary.csv'
    per_run_summary_df.to_csv(per_run_summary_path, index=False)
    print(f'Saved summary CSV: {summary_path}')
    print(f'Saved per-run summary CSV: {per_run_summary_path}')

    fig = make_summary_plot(summary_df)
    out = OUT_DIR / '20260320_mse_v1_summary.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved summary figure: {out}')
    plt.close(fig)

    fig = make_straightness_plot(flat_frames, summary_df)
    out = OUT_DIR / '20260320_mse_v1_straightness.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved straightness figure: {out}')
    plt.close(fig)

    fig = make_tracking_paper_plot(nested_frames, summary_df)
    out = OUT_DIR / '20260320_mse_v1_tracking_paper.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved tracking figure: {out}')
    plt.close(fig)

    fig = make_lateral_proxy_plot(nested_frames, summary_df)
    out = OUT_DIR / '20260320_mse_v1_lateral_proxy.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'Saved lateral proxy figure: {out}')
    plt.close(fig)

    print('\nNote: summary bars are computed by first summarizing each run separately, then averaging those run-level statistics across 320_4~320_6.')


if __name__ == '__main__':
    main()
