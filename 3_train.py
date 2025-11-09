import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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

    #model = Model(in_seq_len=240, out_seq_len=48, in_feat_size = 3, out_feat_size = 1, hidden_feat_size = 256).to(device)
    model = Model(cfg).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    # lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
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

    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"=> loading checkpoint '{cfg.resume}'")
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
    global_step = 0

    for epoch_i in range(start_epoch, cfg.epochs):
        model.train()

        loss_sum = 0
        loss_sum_epoch = 0   # 统计整个 epoch 的累计 loss

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_i} [Train]", ncols=100)
        for train_i, (history_data, future_power) in enumerate(pbar):
            history_data = history_data.to(device)
            future_power = future_power.to(device)

            optimizer.zero_grad()

            predicted_power = model(history_data, future_power.unsqueeze(2))  # Transformer
            loss = loss_function(predicted_power, future_power.unsqueeze(2))  # Transformer

            batch_loss = loss.item()
            # loss_sum += batch_loss
            loss_sum_epoch += batch_loss

            # if (train_i+1) % 100 == 0:
            #     print(f"Epoch: {epoch_i}, Step: {train_i}, Train Loss (last 100): {loss_sum/100:.4f}")
            #     loss_sum = 0   # 清零局部统计

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            pbar.set_postfix(batch_loss=loss.item())

        train_loss = loss_sum_epoch / len(train_dataloader)
        print(f"[Epoch {epoch_i}] Train Loss (avg): {train_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", train_loss, epoch_i)

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

        # 只在 rank=0 打印 & 保存
        print(f"[Epoch {epoch_i}] LR: {lr:.6f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val ACC MAE: {val_acc_mae:.4f} | Val ACC RMSE: {val_acc_rmse:.4f}")
        writer.add_scalar("val_loss", val_loss, epoch_i)
        writer.add_scalar("val_mae", val_mae, epoch_i)
        writer.add_scalar("val_acc_mae", val_acc_mae, epoch_i)
        writer.add_scalar("Val/Acc_RMSE", val_acc_rmse, epoch_i)


        checkpoint = {
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }
        torch.save(checkpoint, f"checkpoint_epoch{epoch_i}.pth")

        if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, f"best_checkpoint_epoch{epoch_i}.pth")
                print(f" Saved checkpoint at epoch {epoch_i} (Val Loss: {val_loss:.4f})")



def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--data-path', type=str, default='data.csv')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--lr-final', type=float, default=0.0001)
    parser.add_argument('--in-seq-len', type=int, default=240)
    parser.add_argument('--out-seq-len', type=int, default=48)
    parser.add_argument('--in-feat-size', type=int, default=3)
    parser.add_argument('--out-feat-size', type=int, default=1)
    parser.add_argument('--hidden-feat-size', type=int, default=256)
    parser.add_argument('--subset', type=float, default=1)
    parser.add_argument('--resume', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_cfg()
    #Power_Dataset(cfg)
    train_val(cfg)