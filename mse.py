import pandas as pd
import torch
import numpy as np

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
A_power_true = data_power[split_idx:]

k = len(A_date)
print(f"训练集 B 长度: {len(B_date)}")
print(f"验证集 A 长度 (K): {k}")

# 归一化（使用训练集B的统计量，确保B和A在同一尺度）
B_date_min, B_date_max = B_date.min(), B_date.max()
B_time_min, B_time_max = B_time.min(), B_time.max()

B_date = (B_date - B_date_min) / (B_date_max - B_date_min + 1e-8)
B_time = (B_time - B_time_min) / (B_time_max - B_time_min + 1e-8)
A_date = (A_date - B_date_min) / (B_date_max - B_date_min + 1e-8)
A_time = (A_time - B_time_min) / (B_time_max - B_time_min + 1e-8)

# ===== 转为 tensor =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B_date_t = torch.tensor(B_date, device=device)
B_time_t = torch.tensor(B_time, device=device)
B_power_t = torch.tensor(B_power, device=device)
A_date_t = torch.tensor(A_date, device=device)
A_time_t = torch.tensor(A_time, device=device)
A_power_t = torch.tensor(A_power_true, device=device)

#检查长度是否足够
if len(B_date_t) < k:
    raise ValueError("训练集长度不足，无法生成与 A 等长的窗口！")

# 构造所有相邻 K 段 B 窗口
num_B = len(B_date) - k + 1
B_date_windows = B_date_t.unfold(0, k, 1)
B_time_windows = B_time_t.unfold(0, k, 1)
B_power_windows = B_power_t.unfold(0, k, 1)

#计算 A 与每个 B 窗口的 MSE
diff_date = B_date_windows - A_date_t.unsqueeze(0)
diff_time = B_time_windows - A_time_t.unsqueeze(0)
mse_all = ((diff_date ** 2) + (diff_time ** 2)).mean(dim=1)

#找最相似的 B 段
best_idx = torch.argmin(mse_all).item()
pred_power = B_power_windows[best_idx]

#计算预测误差
mse = torch.mean((pred_power - A_power_t) ** 2).item()
mae = torch.mean(torch.abs(pred_power - A_power_t)).item()

# 准确率计算（限制在0-1范围）
acc = 1 - mae / (torch.mean(A_power_t) + 1e-6)
acc = max(0.0, min(1.0, acc))

# 相对误差准确率
eps = 1e-6
mask = A_power_t > eps  # 避免除零
if mask.sum() > 0:
    pred_sel = pred_power[mask]
    true_sel = A_power_t[mask]
    denom = true_sel + eps
    acc2 = 1 - torch.sqrt(torch.mean(((pred_sel - true_sel) / denom) ** 2)).item()
    acc2 = max(0.0, min(1.0, acc2))
else:
    acc2 = float('nan')

# ===== 输出结果 =====
print("\n" + "="*50)
print("匹配结果")
print("="*50)
print(f"最相似的历史片段索引: {best_idx}")
print(f"最小时间MSE: {mse_all[best_idx].item():.6f}")
print(f"预测功率 MSE: {mse:.6f}")
print(f"预测功率 MAE: {mae:.6f}")
print(f"预测准确率 ACC (基于MAE): {acc:.6f}")
print(f"预测准确率 ACC2 (基于相对RMSE): {acc2:.6f}")
print("="*50)
