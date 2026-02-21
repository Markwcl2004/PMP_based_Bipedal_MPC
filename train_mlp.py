#!/usr/bin/env python3
"""
MLP状态估计器训练脚本

支持两种训练模式:
1. 直接学习: x_{k+1} = F_NN(x_k, u_k)
2. 残差学习: δ = x_next_true - F_nominal(x_k, u_k)  (推荐)

输入: state(12) + control(6) = 18维
输出: next_state(12) 或 residual(12)
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

class RobotDynamicsDataset(Dataset):
    """机器人动力学数据集 (支持直接学习和残差学习)"""

    def __init__(self, csv_files, residual_mode=True):
        """
        Args:
            csv_files: CSV文件路径列表或单个文件路径
            residual_mode: True=残差学习, False=直接学习
        """
        self.residual_mode = residual_mode

        # 支持单个文件或多个文件
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        # 合并所有数据文件
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f}: {len(df)} samples")

        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(self.data)}")

        # 提取列 (使用新的列名格式)
        # x_n_0..11: 当前状态 (LKF估计, 带噪声)
        # u_n_0..5: 控制输入
        # x_next_est_0..11: 名义模型预测
        # x_next_true_0..11: Mujoco真实下一状态

        state_cols = [f'x_n_{i}' for i in range(12)]
        control_cols = [f'u_n_{i}' for i in range(6)]
        next_est_cols = [f'x_next_est_{i}' for i in range(12)]
        next_true_cols = [f'x_next_true_{i}' for i in range(12)]

        # 检查列是否存在
        if all(col in self.data.columns for col in state_cols):
            self.states = self.data[state_cols].values.astype(np.float32)
            self.controls = self.data[control_cols].values.astype(np.float32)
            self.next_est = self.data[next_est_cols].values.astype(np.float32)
            self.next_true = self.data[next_true_cols].values.astype(np.float32)
        else:
            raise ValueError(f"Required columns not found. Expected: {state_cols}")

        # 计算残差 (仅在残差模式下使用)
        if residual_mode:
            self.residuals = self.next_true - self.next_est
            self.output_data = self.residuals
            print(f"\n[残差学习模式]")
            print(f"  残差均值: {self.residuals.mean(axis=0)}")
            print(f"  残差标准差: {self.residuals.std(axis=0)}")
            print(f"  残差范围: [{self.residuals.min():.4f}, {self.residuals.max():.4f}]")
        else:
            self.output_data = self.next_true
            print(f"\n[直接学习模式]")

        # 计算归一化参数
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-8
        self.control_mean = self.controls.mean(axis=0)
        self.control_std = self.controls.std(axis=0) + 1e-8

        if residual_mode:
            # 残差通常较小，使用残差自身的统计量归一化
            self.output_mean = self.residuals.mean(axis=0)
            self.output_std = self.residuals.std(axis=0) + 1e-8
        else:
            # 直接学习模式使用状态的统计量
            self.output_mean = self.next_true.mean(axis=0)
            self.output_std = self.next_true.std(axis=0) + 1e-8

        # 归一化
        self.states_norm = (self.states - self.state_mean) / self.state_std
        self.controls_norm = (self.controls - self.control_mean) / self.control_std
        self.output_norm = (self.output_data - self.output_mean) / self.output_std

        print(f"\n归一化参数:")
        print(f"  State mean: {self.state_mean}")
        print(f"  State std: {self.state_std}")
        print(f"  Control mean: {self.control_mean}")
        print(f"  Control std: {self.control_std}")
        print(f"  Output mean: {self.output_mean}")
        print(f"  Output std: {self.output_std}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 当前状态和控制输入 (归一化)
        state = torch.FloatTensor(self.states_norm[idx])
        control = torch.FloatTensor(self.controls_norm[idx])

        # 输出 (归一化的残差或下一状态)
        output = torch.FloatTensor(self.output_norm[idx])

        # 拼接输入
        x = torch.cat([state, control])  # 18维
        y = output  # 12维

        return x, y


# ========== MLP网络定义 ==========

class MLPDynamicsModel(nn.Module):
    """MLP动力学模型"""

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

        # 输出层 (线性激活)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ========== 训练函数 ==========

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
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
    """验证"""
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

def export_model_to_txt(model, output_file, dataset, hidden_dims, residual_mode):
    """将模型导出为文本格式 (供C++加载)"""
    model.eval()

    with open(output_file, 'w') as f:
        # 写入网络结构
        input_dim = 18
        output_dim = 12

        f.write(f"{input_dim} {output_dim} {len(hidden_dims)}\n")
        for dim in hidden_dims:
            f.write(f"{dim} ")
        f.write("\n")

        # 写入权重和偏置
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                data = param.detach().cpu().numpy()

                if 'weight' in name:
                    # 权重矩阵 (每行)
                    for row in data:
                        f.write(" ".join(map(str, row)) + "\n")
                elif 'bias' in name:
                    # 偏置向量
                    f.write(" ".join(map(str, data)) + "\n")

    print(f"Model exported to {output_file}")

    # 生成YAML配置文件
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': 'relu',
        'residual_mode': residual_mode,
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
    """分析残差分布"""
    if not dataset.residual_mode:
        print("跳过残差分析 (非残差模式)")
        return

    residuals = dataset.residuals

    # 各维度的统计
    print("\n========== 残差分析 ==========")
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

    # 绘制残差分布直方图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, state_names)):
        ax.hist(residuals[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', label='zero')
        ax.set_title(f'{name}\nmean={residuals[:, i].mean():.4f}, std={residuals[:, i].std():.4f}')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')

    plt.tight_layout()
    output_path = Path(output_dir) / 'residual_distribution.png'
    plt.savefig(output_path)
    print(f"\n残差分布图保存至: {output_path}")
    plt.close()


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='Train MLP dynamics model (supports residual learning)')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path to training data CSV file(s), supports glob patterns')
    parser.add_argument('--output', type=str, default='models/mlp_residual_model.txt',
                       help='Output model file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32],
                       help='Hidden layer dimensions (default: 64 32 for residual)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--no-residual', action='store_true',
                       help='Use direct learning instead of residual learning')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze residuals, do not train')

    args = parser.parse_args()

    residual_mode = not args.no_residual

    # 展开glob模式
    data_files = []
    for pattern in args.data:
        expanded = glob.glob(pattern)
        if expanded:
            data_files.extend(expanded)
        else:
            data_files.append(pattern)

    print("=" * 60)
    print("MLP 动力学模型训练")
    print("=" * 60)
    print(f"模式: {'残差学习' if residual_mode else '直接学习'}")
    print(f"数据文件: {data_files}")
    print(f"输出文件: {args.output}")
    print(f"隐藏层: {args.hidden_dims}")
    print(f"学习率: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 加载数据集
    print(f"\nLoading dataset...")
    dataset = RobotDynamicsDataset(data_files, residual_mode=residual_mode)

    # 分析残差
    analyze_residuals(dataset, output_dir)

    if args.analyze_only:
        print("\n仅分析模式，跳过训练")
        return

    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    print(f"\nTrain samples: {train_size}, Val samples: {val_size}")

    # 创建模型
    model = MLPDynamicsModel(input_dim=18,
                            hidden_dims=args.hidden_dims,
                            output_dim=12,
                            dropout=args.dropout).to(device)

    print(f"\nModel architecture:")
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     patience=10, factor=0.5)

    # 训练循环
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

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'best_model.pth'))
            print(f"  -> New best model (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training Curve ({"Residual" if residual_mode else "Direct"} Learning)')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(output_dir / 'training_curve.png'))
    print(f"\nTraining curve saved to {output_dir / 'training_curve.png'}")
    plt.close()

    # 导出模型
    model.load_state_dict(torch.load(str(output_dir / 'best_model.pth')))
    export_model_to_txt(model, args.output, dataset, args.hidden_dims, residual_mode)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型文件: {args.output}")
    print(f"配置文件: {args.output.replace('.txt', '.yaml')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
