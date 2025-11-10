"""
读取并打印 TensorBoard 日志的简单脚本
"""
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

def read_tensorboard_logs(logdir):
    """读取 TensorBoard 日志"""
    # 找到所有实验目录
    exp_dirs = glob.glob(os.path.join(logdir, "exp_*"))

    if not exp_dirs:
        print(f"在 {logdir} 中没有找到实验目录")
        return

    for exp_dir in sorted(exp_dirs):
        print(f"\n{'='*60}")
        print(f"实验目录: {exp_dir}")
        print(f"{'='*60}")

        # 加载事件文件
        ea = event_accumulator.EventAccumulator(exp_dir)
        ea.Reload()

        # 获取所有标量标签
        tags = ea.Tags()['scalars']
        print(f"\n记录的指标: {tags}")

        # 打印每个指标的最新值
        print("\n最新指标值:")
        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                print(f"  {tag:30s}: {latest.value:.6f} (epoch {latest.step})")

        # 打印训练历史摘要
        if 'Loss/train' in tags and 'Loss/val' in tags:
            train_loss = ea.Scalars('Loss/train')
            val_loss = ea.Scalars('Loss/val')

            print(f"\n训练摘要:")
            print(f"  总轮数: {len(train_loss)}")
            print(f"  初始训练损失: {train_loss[0].value:.6f}")
            print(f"  最终训练损失: {train_loss[-1].value:.6f}")
            print(f"  初始验证损失: {val_loss[0].value:.6f}")
            print(f"  最终验证损失: {val_loss[-1].value:.6f}")
            print(f"  最佳验证损失: {min(e.value for e in val_loss):.6f}")

if __name__ == '__main__':
    import sys
    logdir = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    read_tensorboard_logs(logdir)
