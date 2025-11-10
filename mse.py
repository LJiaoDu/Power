import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# ===== 配置参数 =====
INPUT_LEN = 240   # 输入序列长度（20小时 * 12）
OUTPUT_LEN = 48   # 输出序列长度（4小时 * 12）
TOTAL_LEN = INPUT_LEN + OUTPUT_LEN  # 288

#读取 data.csv
data = pd.read_csv("data.csv", header=None, names=["date", "time", "power"])
data_date = data["date"].values.astype(np.float32)
data_time = data["time"].values.astype(np.float32)
data_power = data["power"].values.astype(np.float32)

#分割 B(训练) 和 A(验证)
split_idx = 95328
B_date = data_date[:split_idx]
B_time = data_time[:split_idx]
B_power = data_power[:split_idx]

A_date = data_date[split_idx:]
A_time = data_time[split_idx:]
A_power = data_power[split_idx:]

print(f"训练集 B 长度: {len(B_date)}")
print(f"验证集 A 长度: {len(A_date)}")

# 归一化（使用训练集B的统计量，确保B和A在同一尺度）
B_date_min, B_date_max = B_date.min(), B_date.max()
B_time_min, B_time_max = B_time.min(), B_time.max()

B_date = (B_date - B_date_min) / (B_date_max - B_date_min + 1e-8)
B_time = (B_time - B_time_min) / (B_time_max - B_time_min + 1e-8)
A_date = (A_date - B_date_min) / (B_date_max - B_date_min + 1e-8)
A_time = (A_time - B_time_min) / (B_time_max - B_time_min + 1e-8)

# ===== 转为 tensor =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

B_date_t = torch.tensor(B_date, device=device)
B_time_t = torch.tensor(B_time, device=device)
B_power_t = torch.tensor(B_power, device=device)
A_date_t = torch.tensor(A_date, device=device)
A_time_t = torch.tensor(A_time, device=device)
A_power_t = torch.tensor(A_power, device=device)

# ===== 构造训练集 B 的所有窗口 =====
# 每个窗口包含：240点输入 + 48点输出
# 可用窗口数量：len(B) - TOTAL_LEN + 1
if len(B_date_t) < TOTAL_LEN:
    raise ValueError(f"训练集长度不足！需要至少 {TOTAL_LEN} 个点")

num_B_windows = len(B_date_t) - TOTAL_LEN + 1
print(f"训练集可用窗口数: {num_B_windows}")

# 使用 unfold 构造滑动窗口
B_date_input = B_date_t.unfold(0, INPUT_LEN, 1)[:num_B_windows]  # [num_windows, 240]
B_time_input = B_time_t.unfold(0, INPUT_LEN, 1)[:num_B_windows]  # [num_windows, 240]
B_power_output = B_power_t.unfold(0, OUTPUT_LEN, 1)[INPUT_LEN:INPUT_LEN+num_B_windows]  # [num_windows, 48]

# ===== 构造验证集 A 的所有样本 =====
if len(A_date_t) < TOTAL_LEN:
    raise ValueError(f"验证集长度不足！需要至少 {TOTAL_LEN} 个点")

num_A_samples = len(A_date_t) - TOTAL_LEN + 1
print(f"验证集样本数: {num_A_samples}")

# ===== 对每个验证样本进行预测 =====
all_mse = []
all_mae = []
all_predictions = []
all_targets = []

print("\n开始预测...")
for sample_idx in tqdm(range(num_A_samples), desc="预测进度"):
    # 提取当前验证样本
    A_input_date = A_date_t[sample_idx : sample_idx + INPUT_LEN]  # [240]
    A_input_time = A_time_t[sample_idx : sample_idx + INPUT_LEN]  # [240]
    A_target_power = A_power_t[sample_idx + INPUT_LEN : sample_idx + TOTAL_LEN]  # [48]

    # 计算该样本与所有训练窗口的时间相似度（MSE）
    diff_date = B_date_input - A_input_date.unsqueeze(0)  # [num_windows, 240]
    diff_time = B_time_input - A_input_time.unsqueeze(0)  # [num_windows, 240]
    mse_time = ((diff_date ** 2) + (diff_time ** 2)).mean(dim=1)  # [num_windows]

    # 找到最相似的训练窗口
    best_idx = torch.argmin(mse_time).item()

    # 用该窗口的输出作为预测
    pred_power = B_power_output[best_idx]  # [48]

    # 计算该样本的误差
    sample_mse = torch.mean((pred_power - A_target_power) ** 2).item()
    sample_mae = torch.mean(torch.abs(pred_power - A_target_power)).item()

    all_mse.append(sample_mse)
    all_mae.append(sample_mae)
    all_predictions.append(pred_power.cpu().numpy())
    all_targets.append(A_target_power.cpu().numpy())

# ===== 计算总体指标 =====
all_predictions = torch.tensor(np.array(all_predictions), device=device)  # [num_samples, 48]
all_targets = torch.tensor(np.array(all_targets), device=device)  # [num_samples, 48]

# 全局指标
val_mse = np.mean(all_mse)
val_mae = np.mean(all_mae)

# ACC (基于MAE)
mean_power = torch.mean(all_targets).item()
val_acc_mae = 1 - val_mae / (mean_power + 1e-6)
val_acc_mae = max(0.0, min(1.0, val_acc_mae))

# ACC2 (基于相对RMSE)
eps = 1e-6
mask = all_targets > eps
if mask.sum() > 0:
    pred_sel = all_predictions[mask]
    true_sel = all_targets[mask]
    denom = true_sel + eps
    val_acc_rmse = 1 - torch.sqrt(torch.mean(((pred_sel - true_sel) / denom) ** 2)).item()
    val_acc_rmse = max(0.0, min(1.0, val_acc_rmse))
else:
    val_acc_rmse = float('nan')

# ===== 输出结果 =====
print("\n" + "="*60)
print("基于历史模式匹配的预测结果")
print("="*60)
print(f"验证样本数量: {num_A_samples}")
print(f"每个样本: 输入240点 → 预测48点")
print("-"*60)
print(f"验证集 MSE: {val_mse:.6f}")
print(f"验证集 MAE: {val_mae:.6f}")
print(f"验证集 ACC (基于MAE): {val_acc_mae:.6f}")
print(f"验证集 ACC (基于相对RMSE): {val_acc_rmse:.6f}")
print("="*60)
print("\n提示: 这些指标可以与 3_train.py 训练的 Transformer 模型对比")
