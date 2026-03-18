#!/usr/bin/env python3
"""
MLP残差动力学训练脚本 v2 — 老师方案

核心思想 (无累积误差):
  每个训练样本都以 MuJoCo ground truth 状态为输入:
    输入:  (x_true(k), u(k))             — 18维
    目标:  x_true(k+1) - F_nominal(x_true(k), u(k))  — 12维 (纯模型残差)

  这样 MLP 学到的是 **纯粹的名义模型误差**，不混入 LKF 估计噪声。
  每个样本是独立的一步预测，训练数据没有累积误差。

数据格式要求 (CSV列):
  x_true_0..11    : 当前 MuJoCo ground truth 状态 (12维)
  u_n_0..5        : 当前控制输入 (6维)
  x_next_nom_0..11: F_nominal(x_true_k, u_k) 名义模型从真实状态出发的预测 (12维)
  x_next_true_0..11: 下一步 MuJoCo ground truth 状态 (12维)

与 v1 (train_mlp.py) 的区别:
  v1 输入 = x_n (LKF估计, 带噪声),  残差 = x_next_true - F_nominal(x_n, u)
  v2 输入 = x_true (MuJoCo真值),    残差 = x_next_true - F_nominal(x_true, u)
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

class TrueStateDynamicsDataset(Dataset):
    """基于真实状态的动力学残差数据集 (老师方案)"""

    def __init__(self, csv_files):
        """
        Args:
            csv_files: CSV文件路径列表或单个文件路径
        """
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f}: {len(df)} samples")

        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(self.data)}")

        # ---- 列名定义 ----
        # x_true_0..11  : 当前真实状态 (MuJoCo ground truth)
        # u_n_0..5      : 控制输入
        # x_next_nom_0..11 : F_nominal(x_true, u) 名义模型预测
        # x_next_true_0..11: 下一步真实状态

        true_state_cols = [f'x_true_{i}' for i in range(12)]
        control_cols = [f'u_n_{i}' for i in range(6)]
        next_nom_cols = [f'x_next_nom_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]

        # 检查列是否存在
        missing = [c for c in true_state_cols + control_cols + next_nom_cols + next_true_cols
                   if c not in self.data.columns]
        if missing:
            raise ValueError(
                f"数据缺少以下列: {missing}\n"
                f"v2 训练脚本需要 x_true_*, u_n_*, x_next_nom_*, x_next_true_* 列。\n"
                f"请确保数据采集端记录了 MuJoCo ground truth 状态和基于真实状态的名义预测。"
            )

        self.states = self.data[true_state_cols].values.astype(np.float32)
        self.controls = self.data[control_cols].values.astype(np.float32)
        self.next_nom = self.data[next_nom_cols].values.astype(np.float32)
        self.next_true = self.data[next_true_cols].values.astype(np.float32)

        # 残差 = x_true(k+1) - F_nominal(x_true(k), u(k))
        self.residuals = self.next_true - self.next_nom

        print(f"\n[v2 残差学习 — 老师方案]")
        print(f"  输入: x_true(k) + u(k) = 18维")
        print(f"  目标: x_true(k+1) - F_nominal(x_true(k), u(k)) = 12维")
        print(f"  残差均值: {self.residuals.mean(axis=0)}")
        print(f"  残差标准差: {self.residuals.std(axis=0)}")
        print(f"  残差范围: [{self.residuals.min():.6f}, {self.residuals.max():.6f}]")

        # ---- 归一化参数 ----
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-8
        self.control_mean = self.controls.mean(axis=0)
        self.control_std = self.controls.std(axis=0) + 1e-8
        self.output_mean = self.residuals.mean(axis=0)
        self.output_std = self.residuals.std(axis=0) + 1e-8

        # 归一化
        self.states_norm = (self.states - self.state_mean) / self.state_std
        self.controls_norm = (self.controls - self.control_mean) / self.control_std
        self.output_norm = (self.residuals - self.output_mean) / self.output_std

        print(f"\n归一化参数:")
        print(f"  State mean:   {self.state_mean}")
        print(f"  State std:    {self.state_std}")
        print(f"  Control mean: {self.control_mean}")
        print(f"  Control std:  {self.control_std}")
        print(f"  Output mean:  {self.output_mean}")
        print(f"  Output std:   {self.output_std}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.states_norm[idx])
        control = torch.FloatTensor(self.controls_norm[idx])
        output = torch.FloatTensor(self.output_norm[idx])

        x = torch.cat([state, control])  # 18维
        y = output                       # 12维
        return x, y


# ========== MLP网络定义 ==========

class MLPDynamicsModel(nn.Module):
    """MLP动力学残差模型"""

    def __init__(self, input_dim=18, hidden_dims=[128, 64, 32], output_dim=12, dropout=0.2):
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
    """将模型导出为文本格式 (供C++加载)"""
    model.eval()

    with open(output_file, 'w') as f:
        input_dim = 18
        output_dim = 12

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

    # YAML 配置文件
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': 'relu',
        'residual_mode': True,
        'version': 'v2_true_state',
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

    print("\n========== 残差分析 (x_true 方案) ==========")
    state_names = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                   'rpy_0', 'rpy_1', 'rpy_2', 'omega_x', 'omega_y', 'omega_z']

    print(f"{'维度':<10} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print("-" * 60)
    for i, name in enumerate(state_names):
        mean = residuals[:, i].mean()
        std = residuals[:, i].std()
        min_val = residuals[:, i].min()
        max_val = residuals[:, i].max()
        print(f"{name:<10} {mean:>12.6f} {std:>12.6f} {min_val:>12.6f} {max_val:>12.6f}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.hist(residuals[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', label='zero')
        ax.set_title(f'{name}\nmean={residuals[:, i].mean():.4f}, std={residuals[:, i].std():.4f}')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')

    plt.suptitle('Residual Distribution (v2: true state input)', fontsize=14)
    plt.tight_layout()
    output_path = Path(output_dir) / 'residual_distribution_v2.png'
    plt.savefig(output_path)
    print(f"\n残差分布图保存至: {output_path}")
    plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(
        description='Train MLP residual model v2 (true state input, 老师方案)')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path to training data CSV file(s), supports glob patterns')
    parser.add_argument('--output', type=str, default='models/mlp_v2_model.txt',
                       help='Output model file')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze residuals, do not train')

    args = parser.parse_args()

    # 展开 glob 模式
    data_files = []
    for pattern in args.data:
        expanded = glob.glob(pattern)
        if expanded:
            data_files.extend(expanded)
        else:
            data_files.append(pattern)

    print("=" * 60)
    print("MLP 残差动力学训练 v2 (老师方案: true state input)")
    print("=" * 60)
    print(f"  输入: x_true(k) + u(k)  →  18维")
    print(f"  目标: x_true(k+1) - F_nominal(x_true(k), u(k))  →  12维")
    print(f"  数据文件: {data_files}")
    print(f"  输出文件: {args.output}")
    print(f"  隐藏层: {args.hidden_dims}")
    print(f"  学习率: {args.lr}, Dropout: {args.dropout}")
    print("=" * 60)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print(f"\nLoading dataset...")
    dataset = TrueStateDynamicsDataset(data_files)

    analyze_residuals(dataset, output_dir)

    if args.analyze_only:
        print("\n仅分析模式，跳过训练")
        return

    # 划分 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    print(f"\nTrain samples: {train_size}, Val samples: {val_size}")

    model = MLPDynamicsModel(input_dim=18,
                            hidden_dims=args.hidden_dims,
                            output_dim=12,
                            dropout=args.dropout).to(device)

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # MSE Loss + Adam
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=10, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20

    print("\nStarting training...")
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
            torch.save(model.state_dict(), str(output_dir / 'best_model_v2.pth'))
            print(f"  -> New best model (val_loss={val_loss:.6f})")
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
    plt.title('Training Curve (v2: true state residual)')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / 'training_curve_v2.png'))
    print(f"\nTraining curve saved to {output_dir / 'training_curve_v2.png'}")
    plt.close()

    # 导出模型
    model.load_state_dict(torch.load(str(output_dir / 'best_model_v2.pth')))
    export_model_to_txt(model, args.output, dataset, args.hidden_dims)

    print("\n" + "=" * 60)
    print("训练完成! (v2 — 老师方案)")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型文件: {args.output}")
    print(f"配置文件: {args.output.replace('.txt', '.yaml')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
