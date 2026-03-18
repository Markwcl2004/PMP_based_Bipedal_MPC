#!/usr/bin/env python3
"""
MLP 窗口数据训练脚本

从 MPC 预测窗口数据训练 MLP 残差模型。
数据来自 ConvexMPC::collectWindowData() 采集的完整预测轨迹。

CSV 格式: window_id, step_k, x_pred(12), u(6), x_next_est(12), x_next_true(12)

支持:
- step_range: 选择使用哪些 k 步的数据 ("0" / "0-2" / "0-4" / "all")
- label_mode: "per-step" (每步用自己的残差) / "broadcast-k0" (用 k=0 的残差广播到所有步)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob


# ========== 数据集类 ==========

class RobotWindowDataset(Dataset):
    """MPC 窗口数据集"""

    def __init__(self, csv_files, step_range="0", label_mode="per-step"):
        """
        Args:
            csv_files: CSV 文件路径列表
            step_range: "0" / "0-2" / "0-4" / "all" — 使用哪些 step_k
            label_mode: "per-step" / "broadcast-k0"
        """
        self.label_mode = label_mode

        if isinstance(csv_files, str):
            csv_files = [csv_files]

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f}: {len(df)} rows")

        data = pd.concat(dfs, ignore_index=True)
        print(f"Total rows (all steps): {len(data)}")

        # 解析 step_range
        if step_range == "all":
            k_min, k_max = 0, data['step_k'].max()
        elif '-' in step_range:
            parts = step_range.split('-')
            k_min, k_max = int(parts[0]), int(parts[1])
        else:
            k_min = k_max = int(step_range)

        print(f"Step range: k={k_min}..{k_max}")

        # 提取列名
        x_pred_cols = [f'x_pred_{i}' for i in range(12)]
        u_cols = [f'u_{i}' for i in range(6)]
        x_next_est_cols = [f'x_next_est_{i}' for i in range(12)]
        x_next_true_cols = [f'x_next_true_{i}' for i in range(12)]

        if label_mode == "broadcast-k0":
            # 计算每个窗口 k=0 的残差，广播到该窗口所有步
            k0_data = data[data['step_k'] == 0].copy()
            k0_residuals = (k0_data[x_next_true_cols].values -
                            k0_data[x_next_est_cols].values).astype(np.float32)

            # 构建 window_id -> k0 residual 映射
            k0_res_map = {}
            for idx, row in enumerate(k0_data.itertuples()):
                k0_res_map[row.window_id] = k0_residuals[idx]

            # 过滤 step_range
            filtered = data[(data['step_k'] >= k_min) & (data['step_k'] <= k_max)].copy()
            print(f"Filtered rows (k={k_min}..{k_max}): {len(filtered)}")

            self.states = filtered[x_pred_cols].values.astype(np.float32)
            self.controls = filtered[u_cols].values.astype(np.float32)

            # 用 k=0 残差作为所有步的标签
            residuals = np.zeros((len(filtered), 12), dtype=np.float32)
            for i, row in enumerate(filtered.itertuples()):
                wid = row.window_id
                if wid in k0_res_map:
                    residuals[i] = k0_res_map[wid]
            self.residuals = residuals

        else:  # per-step
            filtered = data[(data['step_k'] >= k_min) & (data['step_k'] <= k_max)].copy()
            print(f"Filtered rows (k={k_min}..{k_max}): {len(filtered)}")

            self.states = filtered[x_pred_cols].values.astype(np.float32)
            self.controls = filtered[u_cols].values.astype(np.float32)
            x_next_est = filtered[x_next_est_cols].values.astype(np.float32)
            x_next_true = filtered[x_next_true_cols].values.astype(np.float32)
            self.residuals = x_next_true - x_next_est

        self.output_data = self.residuals
        print(f"\nSamples: {len(self.states)}")
        print(f"Label mode: {label_mode}")
        print(f"Residual mean: {self.residuals.mean(axis=0)}")
        print(f"Residual std:  {self.residuals.std(axis=0)}")
        print(f"Residual range: [{self.residuals.min():.6f}, {self.residuals.max():.6f}]")

        # 归一化参数
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-8
        self.control_mean = self.controls.mean(axis=0)
        self.control_std = self.controls.std(axis=0) + 1e-8
        self.output_mean = self.residuals.mean(axis=0)
        self.output_std = self.residuals.std(axis=0) + 1e-8

        # 归一化
        self.states_norm = (self.states - self.state_mean) / self.state_std
        self.controls_norm = (self.controls - self.control_mean) / self.control_std
        self.output_norm = (self.output_data - self.output_mean) / self.output_std

        print(f"\nNormalization params:")
        print(f"  State mean: {self.state_mean}")
        print(f"  State std:  {self.state_std}")
        print(f"  Control mean: {self.control_mean}")
        print(f"  Control std:  {self.control_std}")
        print(f"  Output mean: {self.output_mean}")
        print(f"  Output std:  {self.output_std}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.states_norm[idx])
        control = torch.FloatTensor(self.controls_norm[idx])
        output = torch.FloatTensor(self.output_norm[idx])
        x = torch.cat([state, control])  # 18 dim
        return x, output


# ========== MLP 网络定义 ==========

class MLPDynamicsModel(nn.Module):
    """MLP 动力学模型 (与 train_mlp.py 兼容)"""

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

def export_model_to_txt(model, output_file, dataset, hidden_dims):
    """导出模型为文本格式 (与 train_mlp.py 相同格式, C++ MLPNetwork 直接加载)"""
    model.eval()
    input_dim = 18
    output_dim = 12

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

    # YAML 配置
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': 'relu',
        'residual_mode': True,
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

    yaml_file = output_file.replace('.txt', '.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config exported to {yaml_file}")


# ========== 残差分析 ==========

def analyze_residuals(dataset, output_dir):
    residuals = dataset.residuals
    state_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                   'rpy_0', 'rpy_1', 'rpy_2', 'omega_x', 'omega_y', 'omega_z']

    print("\n========== Residual Analysis ==========")
    print(f"{'Dim':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 60)
    for i, name in enumerate(state_names):
        r = residuals[:, i]
        print(f"{name:<10} {r.mean():>12.6f} {r.std():>12.6f} {r.min():>12.6f} {r.max():>12.6f}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.hist(residuals[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title(f'{name}\nmean={residuals[:, i].mean():.4f}, std={residuals[:, i].std():.4f}')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
    plt.tight_layout()
    output_path = Path(output_dir) / 'window_residual_distribution.png'
    plt.savefig(output_path)
    print(f"\nResidual distribution saved to: {output_path}")
    plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(
        description='Train MLP from MPC window data (supports multi-step training)')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                        help='Window data CSV file(s), supports glob')
    parser.add_argument('--output', type=str, default='models/mlp_window_model.txt',
                        help='Output model file')
    parser.add_argument('--step-range', type=str, default='0',
                        help='Step range: "0", "0-2", "0-4", "all"')
    parser.add_argument('--label-mode', type=str, default='per-step',
                        choices=['per-step', 'broadcast-k0'],
                        help='Label mode: per-step or broadcast-k0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze residuals, skip training')

    args = parser.parse_args()

    # 展开 glob
    data_files = []
    for pattern in args.data:
        expanded = glob.glob(pattern)
        if expanded:
            data_files.extend(expanded)
        else:
            data_files.append(pattern)

    print("=" * 60)
    print("MLP Window Data Training")
    print("=" * 60)
    print(f"Data files:  {data_files}")
    print(f"Output:      {args.output}")
    print(f"Step range:  {args.step_range}")
    print(f"Label mode:  {args.label_mode}")
    print(f"Hidden dims: {args.hidden_dims}")
    print(f"LR:          {args.lr}")
    print(f"Dropout:     {args.dropout}")
    print("=" * 60)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 加载数据
    print(f"\nLoading window dataset...")
    dataset = RobotWindowDataset(data_files,
                                  step_range=args.step_range,
                                  label_mode=args.label_mode)

    analyze_residuals(dataset, output_dir)

    if args.analyze_only:
        print("\nAnalysis-only mode, skipping training")
        return

    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    print(f"\nTrain: {train_size}, Val: {val_size}")

    # 创建模型
    model = MLPDynamicsModel(input_dim=18,
                             hidden_dims=args.hidden_dims,
                             output_dim=12,
                             dropout=args.dropout).to(device)
    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                      patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20

    print("\nTraining...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs} - "
              f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'best_window_model.pth'))
            print(f"  -> New best (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Window Training (step={args.step_range}, label={args.label_mode})')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / 'window_training_curve.png'))
    print(f"\nTraining curve saved to {output_dir / 'window_training_curve.png'}")
    plt.close()

    # 导出
    model.load_state_dict(torch.load(str(output_dir / 'best_window_model.pth')))
    export_model_to_txt(model, args.output, dataset, args.hidden_dims)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model: {args.output}")
    print(f"Config: {args.output.replace('.txt', '.yaml')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
