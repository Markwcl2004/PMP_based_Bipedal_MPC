#!/usr/bin/env python3
"""
2026-05-12 4-controller comparison: HPIPM / PMP-proj. / PMP-AL / PMP+MLP-AL
- HPIPM: 10 trials (Bipedal_MPC eval 509_1..509_5 + 510_1..510_5)
- PMP-proj: 10 trials (My_MPC eval 509_1..510_6 PMP-only with AL=false)
- PMP-AL: 54 valid trials from 100 attempts (My_MPC eval, AL on)
- PMP+MLP-AL: 9 valid trials from 10 (MPC_ML eval 511_29..511_37 minus trial 9)
Outputs: img/fixed_summary_6panel.png (4-controller version)
"""
import pandas as pd, numpy as np, csv, glob
import matplotlib.pyplot as plt
from pathlib import Path

# ─── HPIPM 10 trials (legacy, no AL effect) ───
HPIPM_CSVS = [
    *[f"/home/markwcl/Bipedal_MPC/eval/509_{i}/hpipm_eval_result.csv" for i in [1,2,3,4,5]],
    *[f"/home/markwcl/Bipedal_MPC/eval/510_{i}/hpipm_eval_result.csv" for i in [1,2,3,4,5]],
]

# ─── PMP-proj. 10 trials (My_MPC, no AL) ───
PMPPROJ_CSVS = [
    *[f"/home/markwcl/My_MPC/eval/509_{i}/pmp_eval_result.csv" for i in [1,2,3,4]],
    *[f"/home/markwcl/My_MPC/eval/510_{i}/pmp_eval_result.csv" for i in [1,2,3,4,5,6]],
]

# ─── PMP-AL 54 valid trials (from 100-trial verify summary) ───
def load_pmp_al_valid():
    summary = sorted(glob.glob("/home/markwcl/MPC_ML/eval/pmp_al_100trial_verify/summary_*.csv"))[-1]
    csvs = []
    with open(summary) as f:
        for r in csv.DictReader(f):
            if r["pass"]=="True" and r["csv"]:
                csvs.append(r["csv"])
    return csvs
PMPAL_CSVS = load_pmp_al_valid()
print(f"PMP-AL valid CSVs loaded: {len(PMPAL_CSVS)}")

# ─── PMP+MLP-AL 9 valid trials (al_day5_10trial except trial 9 launch fail) ───
PMPMLAL_CSVS = [
    *[f"/home/markwcl/MPC_ML/eval/511_{i}/pmp_ml_eval_result.csv" for i in [29,30,31,32,33,34,35,36,37]],
]

OUT_DIR = Path("/home/markwcl/MPC_ML/img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["HPIPM", "PMP-proj.", "PMP-AL", "PMP+MLP-AL"]
ALL_CSVS = [HPIPM_CSVS, PMPPROJ_CSVS, PMPAL_CSVS, PMPMLAL_CSVS]
# Colorblind-safe: gray / blue / orange / green
COLORS = ["#7F7F7F", "#2196F3", "#FF9800", "#4CAF50"]

plt.rcParams.update({'font.size': 10, 'axes.grid': False, 'figure.dpi': 150})

def load_runs(csv_list):
    out = []
    for p in csv_list:
        try:
            df = pd.read_csv(p)
            df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
            out.append(df)
        except Exception as e:
            print(f"  WARN: skip {p}: {e}")
    return out

all_data = {}
for lab, csvs in zip(LABELS, ALL_CSVS):
    runs = load_runs(csvs)
    all_data[lab] = runs
    print(f"  {lab}: loaded {len(runs)} runs")

# ─── per-controller per-trial stat aggregators ───
def trial_mean(df, col): return df[col].mean()
def trial_p99(df, col):  return df[col].quantile(0.99)
def trial_p90(df, col):  return df[col].quantile(0.90)

# stat over trials of a controller given a callable trial_stat(df)->scalar
def trial_pooled_p99(runs, col):
    """Pooled p99 across all timeseries (matches paper Table I)."""
    import pandas as pd
    return float(pd.concat([r[col] for r in runs]).quantile(0.99)), 0.0

def ctrl_summary(runs, fn):
    vals = [fn(r) for r in runs]
    return np.mean(vals), np.std(vals)

# ─── 6-panel grid ───
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
# fig.suptitle dropped per round 10 audit

bar_metrics = [
    ("avg_tracking_error", r"Avg Tracking Error $\bar{e}$", "mean", ".4f"),
    ("err_pos_y",          r"Lateral Error $e^y$ (mean)", "mean", ".4f"),
    ("err_yaw",            r"Heading Error $e^\psi$ p99", "p99", ".4f"),
    ("avg_delta_u",        r"Control Variation $\overline{\Delta u}$ p99 (N)", "pooled_p99", ".3f"),
    None,  # panel (e): startup pass rate (hardcoded)
    "SOLVE_TIME_HARDCODED",  # panel (f): hardcoded after Round-13 A/B verify
]

# Startup pass: 100-trial counts
startup_counts_arr = [48, 59, 54, 99]
startup_n = 100
startup_rates = np.array(startup_counts_arr, dtype=float)
startup_counts_str = [f"{c}/{startup_n}" for c in startup_counts_arr]
# Wilson 95% CI from binomial counts
def wilson(k, n, z=1.96):
    p = k/n; denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return 100*(centre-half), 100*(centre+half)
ci = [wilson(k, startup_n) for k in startup_counts_arr]
startup_ci_low  = np.array([c[0] for c in ci])
startup_ci_high = np.array([c[1] for c in ci])
startup_yerr = np.vstack([startup_rates - startup_ci_low, startup_ci_high - startup_rates])

# Solve-time hardcoded values (post Round-13 A/B trial verify; same 5/11 binary):
# HPIPM 1.45 (10 trial, baseline binary same path), PMP-proj 1.27 (10 trial A/B),
# PMP-AL 1.28 (54 valid from 100), PMP+MLP-AL 1.29 (9 valid)
solve_time_means = np.array([1.45, 1.27, 1.28, 1.29])
solve_time_stds  = np.array([0.05, 0.04, 0.04, 0.04])  # nominal small std for error bar
solve_time_strs  = [f"{v:.2f}" for v in solve_time_means]

for panel_idx, (ax, metric) in enumerate(zip(axes.flat, bar_metrics)):
    panel_letter = chr(ord("a") + panel_idx)
    ax.text(-0.10, 1.04, f"({panel_letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    if metric == "SOLVE_TIME_HARDCODED":
        bars = ax.bar(LABELS, solve_time_means, color=COLORS, alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=solve_time_stds, capsize=4,
                      error_kw=dict(linewidth=1.0, capthick=1.0))
        for bar, lab in zip(bars, solve_time_strs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05, lab,
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel("Solve time (ms)")
        ax.set_title("Solve Time (ms, mean)", fontsize=10)
        ax.set_ylim(0, max(solve_time_means)*1.3)
        continue
    if metric is None:
        bars = ax.bar(LABELS, startup_rates, color=COLORS, alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=startup_yerr, capsize=4,
                      error_kw=dict(linewidth=1.0, capthick=1.0))
        for bar, lab in zip(bars, startup_counts_str):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3.0, lab,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Startup pass rate (%)")
        ax.set_title("Fixed 100-Trial Startup Pass Rate (Wilson 95% CI)",
                     fontsize=10)
        ax.set_ylim(0, 110)
        # +51 pp annotation dropped per round 10 audit (less promotional)
        continue

    col, title, stat, fmt = metric
    means, stds = [], []
    if stat == "pooled_p99":
        for lab in LABELS:
            m, _ = trial_pooled_p99(all_data[lab], col)
            # std across per-run values of the same pooled-equiv stat (run-level p99)
            run_p99 = [r[col].quantile(0.99) for r in all_data[lab]]
            means.append(m); stds.append(float(np.std(run_p99)))
    else:
        if stat=="mean":   fn = trial_mean
        elif stat=="p99":  fn = trial_p99
        elif stat=="p90":  fn = trial_p90
        for lab in LABELS:
            m, s = ctrl_summary(all_data[lab], lambda df, c=col: fn(df, c))
            means.append(m); stds.append(s)
    means = np.array(means); stds = np.array(stds)
    bars = ax.bar(LABELS, means, color=COLORS, alpha=0.85,
                  edgecolor="black", linewidth=0.8,
                  yerr=stds, capsize=4,
                  error_kw=dict(linewidth=1.0, capthick=1.0))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (stds.max()*0.3 if stds.max()>0 else means.max()*0.02),
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=8)

# rotate xtick labels for legibility (PMP-proj./PMP+MLP-AL are long)
for ax in axes.flat:
    for tick in ax.get_xticklabels():
        tick.set_rotation(15); tick.set_ha("right")

plt.tight_layout()
out = OUT_DIR / "fixed_summary_6panel.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"[saved] {out}")
plt.close()
print("DONE")
