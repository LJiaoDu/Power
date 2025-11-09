import torch
from torch.utils.data import Dataset, DataLoader
from dnn_model import TFMModel as Model
from tqdm import tqdm
import argparse
import os
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR



class Power_Dataset(Dataset):
    def __init__(self, cfg, phase='train'):
        super().__init__()
        self.csv_data = []
        with open(cfg.data_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.csv_data.append(row)

        self.csv_data = np.array(self.csv_data, np.float32)

        if cfg.subset < 1.0:
            n = int(len(self.csv_data) * cfg.subset)
            self.csv_data = self.csv_data[:n]

        if phase == 'train':
            self.csv_data = self.csv_data[:95328]
        else:
            self.csv_data = self.csv_data[95328:]

    def __len__(self):
        return len(self.csv_data) - 24 * 12
    
    def __getitem__(self, idx):
        return self.csv_data[idx : idx + 20 * 12], self.csv_data[idx + 20 * 12 : idx + 24 * 12, 2]

    
def train_val(cfg):
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            

    train_data = Power_Dataset(cfg=cfg, phase='train')
    val_data = Power_Dataset(cfg=cfg, phase='val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = Model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略：warmup + cosine annealing
    warmup_epochs = 5
    main_epochs = cfg.epochs - warmup_epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=cfg.lr_final)
        ],
        milestones=[warmup_epochs])
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter(f"runs/exp_{time.strftime('%Y%m%d-%H%M%S')}")
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 10  # 早停的耐心值
    patience_counter = 0

    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"=> loading checkpoint '{cfg.resume}'")
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float("inf"))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    for epoch_i in range(start_epoch, cfg.epochs):
        model.train()
        loss_sum_epoch = 0   # 统计整个 epoch 的累计 loss

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_i} [Train]", ncols=100)
        for train_i, (history_data, future_power) in enumerate(pbar):
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            predicted_power = model(history_data, future_power.unsqueeze(2))
            loss = loss_function(predicted_power, future_power.unsqueeze(2))

            loss_sum_epoch += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            pbar.set_postfix(batch_loss=loss.item())

        train_loss = loss_sum_epoch / len(train_dataloader)
        print(f"[Epoch {epoch_i}] Train Loss (avg): {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch_i)

        scheduler.step()

        # 验证
        model.eval()
        val_loss, val_mae = 0, 0
        val_acc_mae, val_acc_rmse = 0, 0
        valid_batches, valid_batches_mae = 0, 0

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_i} [Val]", ncols=120)
            for history_data, future_power in pbar:
                history_data = history_data.to(device)
                future_power = future_power.to(device)

                predicted_power = model(history_data, future_power.unsqueeze(2))
                loss = loss_function(predicted_power, future_power.unsqueeze(2))
                val_loss += loss.item()
                batch_mae = torch.mean(torch.abs(predicted_power - future_power.unsqueeze(2))).item()
                val_mae += batch_mae
                acc_batch_mae = 1 - batch_mae / (torch.mean(future_power.unsqueeze(2)).item() + 1e-6)
                acc_batch_mae = max(min(acc_batch_mae, 1.0), 0.0)
                val_acc_mae += acc_batch_mae
                valid_batches_mae += 1


                mask = future_power > 1e-6
                if mask.sum() > 0:
                    future_power_sel = future_power[mask]
                    predicted_power_sel = predicted_power.squeeze(-1)[mask]

                    eps = 1e-6
                    denom = future_power_sel + eps

                    acc_batch = 1 - torch.sqrt(torch.mean(
                        ((predicted_power_sel - future_power_sel) / denom) ** 2
                    )).item()

                    val_acc_rmse += acc_batch
                    valid_batches += 1

        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_acc_mae  = val_acc_mae / valid_batches_mae if valid_batches_mae > 0 else float('nan')
        val_acc_rmse = val_acc_rmse / valid_batches if valid_batches > 0 else float('nan')

        # 获取当前学习率
        lr = optimizer.param_groups[0]['lr']

        # 打印和记录指标
        print(f"[Epoch {epoch_i}] LR: {lr:.6f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val ACC MAE: {val_acc_mae:.4f} | Val ACC RMSE: {val_acc_rmse:.4f}")
        writer.add_scalar("Loss/val", val_loss, epoch_i)
        writer.add_scalar("Metric/val_mae", val_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_mae", val_acc_mae, epoch_i)
        writer.add_scalar("Metric/val_acc_rmse", val_acc_rmse, epoch_i)
        writer.add_scalar("LearningRate", lr, epoch_i)


        # 保存最新的检查点
        checkpoint = {
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, "checkpoint_latest.pth")

        # 保存最佳检查点和早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, "best_checkpoint.pth")
            print(f"✓ Saved best checkpoint at epoch {epoch_i} (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")

        # 早停检查
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            break



def parse_cfg():
    parser = argparse.ArgumentParser(description='Train Transformer model for power forecasting')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID or "cpu"')
    parser.add_argument('--data-path', type=str, default='data.csv', help='Path to training data CSV file')
    parser.add_argument('--batch-size', type=int, default=48, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--lr-init', type=float, default=0.001, help='Initial learning rate for AdamW')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='Final learning rate after decay')
    parser.add_argument('--in-seq-len', type=int, default=240, help='Input sequence length')
    parser.add_argument('--out-seq-len', type=int, default=48, help='Output sequence length')
    parser.add_argument('--in-feat-size', type=int, default=3, help='Input feature size')
    parser.add_argument('--out-feat-size', type=int, default=1, help='Output feature size')
    parser.add_argument('--hidden-feat-size', type=int, default=256, help='Hidden feature size')
    parser.add_argument('--subset', type=float, default=1.0, help='Subset of data to use (0-1)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    train_val(cfg)