#!/usr/bin/env python3
"""
MLP残差动力学训练脚本 v3 — DAgger 迭代 + 噪声增强

核心思路:
  1. DAgger 迭代: 在 MLP 影响下持续采集 k=0 真实数据，逐轮合并训练
  2. 噪声增强: 对训练数据添加校准噪声，模拟 k>=1 预测状态的偏移
  3. 可选 broadcast-k0 窗口数据: 用 k=0 残差标注 k>=1 预测状态

解决的问题:
  MLP 递归推理时 k>=1 遇到预测状态 (distribution shift)，
  但只在 k=0 真实状态上训练过。噪声增强 + DAgger 消除这个偏移。

数据格式:
  单步数据 (single-step): 支持 v2 格式 (x_true_*) 和 v1 格式 (x_n_*)
  窗口数据 (window): window_id, step_k, x_pred_*, u_*, x_next_est_*, x_next_true_*

用法:
  # Round 0 (基线 + 噪声增强)
  python3 train_mlp_v3.py \\
      --rounds "round0:/home/markwcl/mpc_data/v2/mlp_training_data_*.csv:single-step:1.0" \\
      --noise-enable --noise-alpha 0.5 --noise-augmentation-ratio 3 \\
      --noise-from-data /home/markwcl/mpc_data/v2/mlp_training_data_*.csv \\
      --output models/mlp_v3_round0.txt

  # Round 1 (DAgger 合并训练)
  python3 train_mlp_v3.py \\
      --rounds \\
          "round0:/home/markwcl/mpc_data/v2/mlp_training_data_*.csv:single-step:1.0" \\
          "round1:/home/markwcl/mpc_data/v3/round1/*.csv:single-step:1.5" \\
      --noise-enable --noise-alpha 0.5 --noise-augmentation-ratio 3 \\
      --noise-from-data /home/markwcl/mpc_data/v2/mlp_training_data_*.csv \\
      --warm-start models/best_model_v3.pth \\
      --output models/mlp_v3_round1.txt
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml
import json
from pathlib import Path
import argparse
from math import sqrt
import matplotlib.pyplot as plt
import glob as glob_module


# ========== 噪声校准 ==========

def calibrate_noise_sigma(csv_files, alpha=0.5, recursive_steps=3):
    """
    从单步数据校准噪声 sigma。

    原理: k 步预测误差约为 sqrt(k) * 单步误差
    noise_sigma = alpha * sqrt(recursive_steps) * residual_std

    Args:
        csv_files: 用于校准的 CSV 文件列表
        alpha: 噪声缩放系数 (0.5=保守, 1.0=激进)
        recursive_steps: 递归预测步数

    Returns:
        noise_sigma: 12 维噪声标准差 (numpy array, float32)
    """
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Auto-detect format
    if 'x_true_0' in data.columns:
        next_nom_cols = [f'x_next_nom_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]
        residuals = (data[next_true_cols].values - data[next_nom_cols].values).astype(np.float32)
    elif 'x_n_0' in data.columns:
        next_est_cols = [f'x_next_est_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]
        residuals = (data[next_true_cols].values - data[next_est_cols].values).astype(np.float32)
    else:
        raise ValueError("CSV 格式无法识别: 需要 x_true_* 或 x_n_* 列")

    residual_std = residuals.std(axis=0)
    noise_sigma = (alpha * sqrt(recursive_steps) * residual_std).astype(np.float32)

    state_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                   'rpy_0', 'rpy_1', 'rpy_2', 'omega_x', 'omega_y', 'omega_z']

    print(f"\n[噪声校准] alpha={alpha}, K={recursive_steps}, "
          f"scale=alpha*sqrt(K)={alpha * sqrt(recursive_steps):.3f}")
    print(f"{'维度':<10} {'残差std':>12} {'噪声sigma':>12}")
    print("-" * 36)
    for i, name in enumerate(state_names):
        print(f"{name:<10} {residual_std[i]:>12.6f} {noise_sigma[i]:>12.6f}")

    return noise_sigma


# ========== 数据加载辅助 ==========

def load_single_step_data(csv_files):
    """
    加载单步训练数据，自动检测 v1/v2 格式。

    Returns:
        states: (N, 12) float32
        controls: (N, 6) float32
        residuals: (N, 12) float32
    """
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"    Loaded {f}: {len(df)} samples")

    data = pd.concat(dfs, ignore_index=True)

    # Auto-detect format
    if 'x_true_0' in data.columns:
        # v2 format: true state input
        state_cols = [f'x_true_{i}' for i in range(12)]
        control_cols = [f'u_n_{i}' for i in range(6)]
        next_nom_cols = [f'x_next_nom_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]
        states = data[state_cols].values.astype(np.float32)
        controls = data[control_cols].values.astype(np.float32)
        residuals = (data[next_true_cols].values - data[next_nom_cols].values).astype(np.float32)
        print(f"    Format: v2 (x_true), {len(data)} samples")
    elif 'x_n_0' in data.columns:
        # v1 format: LKF estimate input
        state_cols = [f'x_n_{i}' for i in range(12)]
        control_cols = [f'u_n_{i}' for i in range(6)]
        next_est_cols = [f'x_next_est_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]
        states = data[state_cols].values.astype(np.float32)
        controls = data[control_cols].values.astype(np.float32)
        residuals = (data[next_true_cols].values - data[next_est_cols].values).astype(np.float32)
        print(f"    Format: v1 (x_n), {len(data)} samples")
    else:
        raise ValueError(
            f"CSV 格式无法识别: 需要 x_true_* 或 x_n_* 列\n"
            f"实际列: {list(data.columns[:10])}...")

    return states, controls, residuals


def load_window_data(csv_files, step_range="0-2", label_mode="broadcast-k0"):
    """
    加载窗口数据。

    Returns:
        states: (N, 12) float32
        controls: (N, 6) float32
        residuals: (N, 12) float32
    """
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"    Loaded window {f}: {len(df)} rows")

    data = pd.concat(dfs, ignore_index=True)

    # Parse step_range
    if step_range == "all":
        k_min, k_max = 0, int(data['step_k'].max())
    elif '-' in step_range:
        parts = step_range.split('-')
        k_min, k_max = int(parts[0]), int(parts[1])
    else:
        k_min = k_max = int(step_range)

    x_pred_cols = [f'x_pred_{i}' for i in range(12)]
    u_cols = [f'u_{i}' for i in range(6)]
    x_next_est_cols = [f'x_next_est_{i}' for i in range(12)]
    x_next_true_cols = [f'x_next_true_{i}' for i in range(12)]

    if label_mode == "broadcast-k0":
        # k=0 残差广播到所有步
        k0_data = data[data['step_k'] == 0].copy()
        k0_residuals = (k0_data[x_next_true_cols].values -
                        k0_data[x_next_est_cols].values).astype(np.float32)

        k0_res_map = {}
        for idx, row in enumerate(k0_data.itertuples()):
            k0_res_map[row.window_id] = k0_residuals[idx]

        filtered = data[(data['step_k'] >= k_min) & (data['step_k'] <= k_max)].copy()
        states = filtered[x_pred_cols].values.astype(np.float32)
        controls = filtered[u_cols].values.astype(np.float32)

        residuals = np.zeros((len(filtered), 12), dtype=np.float32)
        for i, row in enumerate(filtered.itertuples()):
            wid = row.window_id
            if wid in k0_res_map:
                residuals[i] = k0_res_map[wid]
    else:  # per-step
        filtered = data[(data['step_k'] >= k_min) & (data['step_k'] <= k_max)].copy()
        states = filtered[x_pred_cols].values.astype(np.float32)
        controls = filtered[u_cols].values.astype(np.float32)
        x_next_est = filtered[x_next_est_cols].values.astype(np.float32)
        x_next_true = filtered[x_next_true_cols].values.astype(np.float32)
        residuals = x_next_true - x_next_est

    print(f"    Window data: {len(states)} samples (k={k_min}..{k_max}, mode={label_mode})")

    return states, controls, residuals


# ========== 数据集类 ==========

class DAggerAugmentedDataset(Dataset):
    """
    多轮 DAgger 数据 + 噪声增强 + 可选 broadcast-k0 窗口数据

    三种数据来源:
    1. 单步数据 (各轮 DAgger 采集的 k=0 真实残差)
    2. 窗口数据 broadcast-k0 (可选，用 k=0 残差标注 k>=1 预测状态)
    3. 噪声增强 (对单步数据添加校准噪声，模拟 k>=1 分布)

    索引布局:
      [0, n_round)                          → 真实 round 样本
      [n_round, n_round + n_window)         → 真实 window 样本
      [n_real, n_real + n_augmented)        → round 样本的噪声副本
    其中 n_real = n_round + n_window
    """

    def __init__(self, round_states, round_controls, round_residuals, round_weights,
                 window_states=None, window_controls=None, window_residuals=None,
                 window_weight=0.5,
                 noise_sigma=None, augmentation_ratio=0):
        """
        Args:
            round_states: list of (N_i, 12) arrays, one per round
            round_controls: list of (N_i, 6) arrays
            round_residuals: list of (N_i, 12) arrays
            round_weights: list of floats, per-round sampling weights
            window_states: optional (M, 12) array
            window_controls: optional (M, 6) array
            window_residuals: optional (M, 12) array
            window_weight: sampling weight for window data
            noise_sigma: (12,) noise std for augmentation, None = disabled
            augmentation_ratio: noisy copies per real round sample (0 = disabled)
        """
        # ---- Merge round data ----
        all_states = []
        all_controls = []
        all_residuals = []
        sample_weights = []

        for states, controls, residuals, weight in zip(
                round_states, round_controls, round_residuals, round_weights):
            n = len(states)
            all_states.append(states)
            all_controls.append(controls)
            all_residuals.append(residuals)
            sample_weights.extend([weight] * n)

        self.n_round_samples = sum(len(s) for s in round_states)

        # ---- Window data ----
        self.n_window_samples = 0
        if window_states is not None and len(window_states) > 0:
            all_states.append(window_states)
            all_controls.append(window_controls)
            all_residuals.append(window_residuals)
            self.n_window_samples = len(window_states)
            sample_weights.extend([window_weight] * self.n_window_samples)

        self.states = np.concatenate(all_states, axis=0)       # (n_real, 12)
        self.controls = np.concatenate(all_controls, axis=0)   # (n_real, 6)
        self.residuals = np.concatenate(all_residuals, axis=0) # (n_real, 12)

        self.n_real = len(self.states)  # n_round + n_window

        # ---- Noise augmentation (only for round data, not window) ----
        self.noise_sigma = noise_sigma.astype(np.float32) if noise_sigma is not None else None
        self.augmentation_ratio = augmentation_ratio if noise_sigma is not None else 0
        self.n_augmented = self.n_round_samples * self.augmentation_ratio

        # Extend weights for augmented samples (same weight as source round sample)
        round_weights_array = sample_weights[:self.n_round_samples]
        for _ in range(self.augmentation_ratio):
            sample_weights.extend(round_weights_array)

        self.sample_weights = np.array(sample_weights, dtype=np.float64)

        total = self.n_real + self.n_augmented
        print(f"\n[DAggerAugmentedDataset]")
        print(f"  Round samples:     {self.n_round_samples}")
        print(f"  Window samples:    {self.n_window_samples}")
        print(f"  Real total:        {self.n_real}")
        print(f"  Augmented copies:  {self.n_augmented} "
              f"({self.augmentation_ratio}x round samples)")
        print(f"  Effective total:   {total}")

        # ---- Compute normalization from all real data (stable, no augmentation noise) ----
        self.state_mean = self.states.mean(axis=0).astype(np.float32)
        self.state_std = (self.states.std(axis=0) + 1e-8).astype(np.float32)
        self.control_mean = self.controls.mean(axis=0).astype(np.float32)
        self.control_std = (self.controls.std(axis=0) + 1e-8).astype(np.float32)
        self.output_mean = self.residuals.mean(axis=0).astype(np.float32)
        self.output_std = (self.residuals.std(axis=0) + 1e-8).astype(np.float32)

        # Pre-normalize controls and residuals (these are not augmented)
        self.controls_norm = ((self.controls - self.control_mean) /
                              self.control_std).astype(np.float32)
        self.residuals_norm = ((self.residuals - self.output_mean) /
                               self.output_std).astype(np.float32)

        print(f"\n归一化参数:")
        print(f"  State mean:   {self.state_mean}")
        print(f"  State std:    {self.state_std}")
        print(f"  Control mean: {self.control_mean}")
        print(f"  Control std:  {self.control_std}")
        print(f"  Output mean:  {self.output_mean}")
        print(f"  Output std:   {self.output_std}")

        if self.noise_sigma is not None:
            print(f"\n噪声增强:")
            print(f"  Noise sigma: {self.noise_sigma}")
            print(f"  Aug ratio:   {self.augmentation_ratio}")

    def __len__(self):
        return self.n_real + self.n_augmented

    def __getitem__(self, idx):
        if idx < self.n_real:
            # Real sample (round or window) — no noise
            state = self.states[idx].copy()
            real_idx = idx
        else:
            # Augmented sample — maps back to a round sample, add noise
            real_idx = (idx - self.n_real) % self.n_round_samples
            state = self.states[real_idx].copy()
            # Add calibrated noise in original space
            state += np.random.randn(12).astype(np.float32) * self.noise_sigma

        # Normalize state (online, since augmented states differ each time)
        state_norm = (state - self.state_mean) / self.state_std

        x = torch.FloatTensor(np.concatenate([state_norm, self.controls_norm[real_idx]]))
        y = torch.FloatTensor(self.residuals_norm[real_idx])
        return x, y


# ========== MLP网络定义 ==========

class MLPDynamicsModel(nn.Module):
    """MLP动力学残差模型 (与 v1/v2 架构和 C++ MLPNetwork 兼容)"""

    def __init__(self, input_dim=18, hidden_dims=[64, 32], output_dim=12, dropout=0.1):
        super(MLPDynamicsModel, self).__init__()
        self.hidden_dims = hidden_dims
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ========== 训练函数 ==========

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ========== 模型导出 ==========

def export_model(model, output_file, dataset, hidden_dims, training_info=None):
    """导出 .txt 权重 + .yaml 配置 (与 C++ MLPNetwork 兼容)"""
    model.eval()
    input_dim = 18
    output_dim = 12

    # ---- .txt weights ----
    with open(output_file, 'w') as f:
        f.write(f"{input_dim} {output_dim} {len(hidden_dims)}\n")
        for dim in hidden_dims:
            f.write(f"{dim} ")
        f.write("\n")

        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                data = param.detach().cpu().numpy()
                if 'weight' in name:
                    for row in data:
                        f.write(" ".join(map(str, row)) + "\n")
                elif 'bias' in name:
                    f.write(" ".join(map(str, data)) + "\n")

    print(f"Model exported to {output_file}")

    # ---- .yaml config ----
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': 'relu',
        'residual_mode': True,
        'version': 'v3_dagger',
        'normalization': {
            'use_input_norm': True,
            'use_output_norm': True,
            'input_mean': dataset.state_mean.tolist() + dataset.control_mean.tolist(),
            'input_std': dataset.state_std.tolist() + dataset.control_std.tolist(),
            'output_mean': dataset.output_mean.tolist(),
            'output_std': dataset.output_std.tolist()
        },
        'model_path': str(output_file)
    }

    if training_info:
        config['training'] = training_info

    yaml_file = output_file.replace('.txt', '.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config exported to {yaml_file}")


# ========== 残差分析 ==========

def analyze_residuals(dataset, output_dir):
    """分析并可视化残差分布"""
    residuals = dataset.residuals
    state_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                   'rpy_0', 'rpy_1', 'rpy_2', 'omega_x', 'omega_y', 'omega_z']

    print("\n========== 残差分析 ==========")
    print(f"{'维度':<10} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print("-" * 60)
    for i, name in enumerate(state_names):
        r = residuals[:, i]
        print(f"{name:<10} {r.mean():>12.6f} {r.std():>12.6f} "
              f"{r.min():>12.6f} {r.max():>12.6f}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.hist(residuals[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title(f'{name}\nmean={residuals[:, i].mean():.4f}, '
                     f'std={residuals[:, i].std():.4f}')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')

    plt.suptitle('Residual Distribution (v3 DAgger)', fontsize=14)
    plt.tight_layout()
    output_path = Path(output_dir) / 'residual_distribution_v3.png'
    plt.savefig(output_path)
    print(f"\n残差分布图保存至: {output_path}")
    plt.close()


# ========== 解析 rounds 参数 ==========

def parse_rounds(round_specs):
    """
    Parse --rounds arguments.
    Format: "name:glob_pattern:type:weight"

    Returns: list of (name, files, type, weight) tuples
    """
    rounds = []
    for spec in round_specs:
        parts = spec.split(':')
        if len(parts) != 4:
            raise ValueError(
                f"Invalid round spec: '{spec}'\n"
                f"Expected format: 'name:glob_pattern:type:weight'\n"
                f"Example: 'round0:/path/to/*.csv:single-step:1.0'")

        name, pattern, dtype, weight = parts
        files = sorted(glob_module.glob(pattern))
        if not files:
            raise ValueError(f"No files found for round '{name}': {pattern}")

        rounds.append((name, files, dtype, float(weight)))
        print(f"  Round '{name}': {len(files)} files, type={dtype}, weight={weight}")

    return rounds


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(
        description='MLP v3 训练: DAgger 迭代 + 噪声增强',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Rounds
    parser.add_argument('--rounds', type=str, nargs='+', required=True,
                        help='"name:glob:type:weight" format, supports multiple rounds')

    # Window data (optional)
    parser.add_argument('--window-data', type=str, nargs='+', default=None,
                        help='Window CSV file(s), supports glob')
    parser.add_argument('--window-step-range', type=str, default='0-2',
                        help='Window step range (default: "0-2")')
    parser.add_argument('--window-label-mode', type=str, default='broadcast-k0',
                        choices=['per-step', 'broadcast-k0'],
                        help='Window label mode (default: broadcast-k0)')
    parser.add_argument('--window-weight', type=float, default=0.5,
                        help='Window data sampling weight (default: 0.5)')

    # Noise augmentation
    parser.add_argument('--noise-enable', action='store_true',
                        help='Enable noise augmentation')
    parser.add_argument('--noise-alpha', type=float, default=0.5,
                        help='Noise scale coefficient (default: 0.5)')
    parser.add_argument('--noise-augmentation-ratio', type=int, default=3,
                        help='Noisy copies per real sample (default: 3)')
    parser.add_argument('--noise-from-data', type=str, nargs='+', default=None,
                        help='CSV files for noise calibration (glob supported)')
    parser.add_argument('--noise-recursive-steps', type=int, default=3,
                        help='Recursive steps for noise calibration (default: 3)')

    # Model
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Path to .pth file for warm-start')
    parser.add_argument('--output', type=str, default='models/mlp_v3_model.txt',
                        help='Output model file (.txt)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--early-stop-patience', type=int, default=20)
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze data, skip training')

    args = parser.parse_args()

    print("=" * 60)
    print("MLP v3 训练: DAgger 迭代 + 噪声增强")
    print("=" * 60)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. Parse and load rounds ----
    print(f"\n[加载 DAgger 轮次数据]")
    rounds = parse_rounds(args.rounds)

    round_states_list = []
    round_controls_list = []
    round_residuals_list = []
    round_weights_list = []
    round_names = []

    for name, files, dtype, weight in rounds:
        if dtype != 'single-step':
            raise ValueError(f"Unsupported round type: {dtype}")

        print(f"\n  Loading round '{name}'...")
        states, controls, residuals = load_single_step_data(files)
        round_states_list.append(states)
        round_controls_list.append(controls)
        round_residuals_list.append(residuals)
        round_weights_list.append(weight)
        round_names.append(name)
        print(f"  {name}: {len(states)} samples, weight={weight}")

    # ---- 2. Load window data (optional) ----
    w_states, w_controls, w_residuals = None, None, None
    if args.window_data:
        print(f"\n[加载窗口数据]")
        window_files = []
        for pattern in args.window_data:
            expanded = sorted(glob_module.glob(pattern))
            if expanded:
                window_files.extend(expanded)
            else:
                window_files.append(pattern)

        w_states, w_controls, w_residuals = load_window_data(
            window_files,
            step_range=args.window_step_range,
            label_mode=args.window_label_mode)

    # ---- 3. Calibrate noise ----
    noise_sigma = None
    aug_ratio = 0
    if args.noise_enable:
        print(f"\n[噪声校准]")
        if args.noise_from_data:
            noise_files = []
            for pattern in args.noise_from_data:
                expanded = sorted(glob_module.glob(pattern))
                if expanded:
                    noise_files.extend(expanded)
                else:
                    noise_files.append(pattern)
        else:
            # Fallback: use round0 data for calibration
            noise_files = rounds[0][1]

        noise_sigma = calibrate_noise_sigma(
            noise_files,
            alpha=args.noise_alpha,
            recursive_steps=args.noise_recursive_steps)
        aug_ratio = args.noise_augmentation_ratio

        # Save noise sigma for reproducibility
        sigma_path = output_dir / 'noise_sigma.json'
        sigma_info = {
            'alpha': args.noise_alpha,
            'recursive_steps': args.noise_recursive_steps,
            'noise_sigma': noise_sigma.tolist()
        }
        with open(sigma_path, 'w') as f:
            json.dump(sigma_info, f, indent=2)
        print(f"  Noise sigma saved to {sigma_path}")

    # ---- 4. Create dataset ----
    print(f"\n[构建数据集]")
    dataset = DAggerAugmentedDataset(
        round_states=round_states_list,
        round_controls=round_controls_list,
        round_residuals=round_residuals_list,
        round_weights=round_weights_list,
        window_states=w_states,
        window_controls=w_controls,
        window_residuals=w_residuals,
        window_weight=args.window_weight,
        noise_sigma=noise_sigma,
        augmentation_ratio=aug_ratio)

    analyze_residuals(dataset, output_dir)

    if args.analyze_only:
        print("\n仅分析模式，跳过训练")
        return

    # ---- 5. Train/val split ----
    # Validation uses only clean real samples (no augmentation noise)
    # Augmented samples that map to val real samples are excluded from training
    n_real = dataset.n_real
    indices = np.arange(n_real)
    np.random.shuffle(indices)
    val_size = int(0.2 * n_real)
    val_indices = indices[:val_size]
    train_real_indices = indices[val_size:]

    # Build train subset: train real indices + augmented indices (from train sources only)
    train_indices = list(train_real_indices)
    train_real_set = set(train_real_indices.tolist())
    for aug_i in range(dataset.n_augmented):
        source_idx = aug_i % dataset.n_round_samples
        if source_idx in train_real_set:
            train_indices.append(dataset.n_real + aug_i)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices.tolist())

    # Weighted sampler for training (per-round weights)
    train_weights = dataset.sample_weights[train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    n_train_aug = len(train_indices) - len(train_real_indices)
    print(f"\nTrain: {len(train_indices)} "
          f"(real: {len(train_real_indices)}, augmented: {n_train_aug})")
    print(f"Val:   {val_size} (real only, no augmentation)")

    # ---- 6. Create model ----
    model = MLPDynamicsModel(input_dim=18,
                             hidden_dims=args.hidden_dims,
                             output_dim=12,
                             dropout=args.dropout).to(device)

    if args.warm_start:
        print(f"\n[Warm-start] Loading weights from {args.warm_start}")
        state_dict = torch.load(args.warm_start, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        print("  Warm-start loaded successfully")

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- 7. Training ----
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_pth = str(output_dir / 'best_model_v3.pth')

    print(f"\nStarting training... (epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr})")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs} - "
              f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_pth)
            print(f"  -> New best model (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # ---- 8. Training curve ----
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Curve (v3 DAgger + Noise Augmentation)')
    plt.legend()
    plt.grid(True)
    curve_path = str(output_dir / 'training_curve_v3.png')
    plt.savefig(curve_path)
    print(f"\nTraining curve saved to {curve_path}")
    plt.close()

    # ---- 9. Export model ----
    model.load_state_dict(torch.load(best_pth, map_location=device,
                                     weights_only=True))

    training_info = {
        'rounds': [{'name': n, 'weight': float(w)}
                    for n, w in zip(round_names, round_weights_list)],
        'noise_augmentation': {
            'enabled': args.noise_enable,
            'alpha': args.noise_alpha if args.noise_enable else None,
            'ratio': aug_ratio,
            'recursive_steps': args.noise_recursive_steps if args.noise_enable else None,
        },
        'warm_start': args.warm_start,
        'best_val_loss': float(best_val_loss),
    }

    export_model(model, args.output, dataset, args.hidden_dims, training_info)

    print("\n" + "=" * 60)
    print("训练完成! (v3 — DAgger + 噪声增强)")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型文件:     {args.output}")
    print(f"配置文件:     {args.output.replace('.txt', '.yaml')}")
    print(f"最佳权重:     {best_pth}")
    print("=" * 60)


if __name__ == '__main__':
    main()
