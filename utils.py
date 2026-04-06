import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed=42):
    """
    固定所有的随机种子，确保每次运行代码时，只要参数相同，结果就完全一样。
    这对于调参和写报告极其重要！
    """
    # 1. 固定 Python 内置 random 模块的随机种子
    random.seed(seed)
    
    # 2. 固定 Numpy 的随机种子
    np.random.seed(seed)
    
    # 3. 固定 PyTorch CPU 的随机种子
    torch.manual_seed(seed)
    
    # 4. 固定 PyTorch GPU 的随机种子 (如果有多个 GPU，用 torch.cuda.manual_seed_all)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 5. 针对 cuDNN (NVIDIA 的深度学习加速库) 的一些设定
    # 牺牲一点点速度，换取 100% 的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"已固定全局随机种子为: {seed}")

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    """
    根据传入的训练历史数据，绘制 Loss 和 Accuracy 曲线并保存。
    
    参数说明:
    - train_losses: 列表，每一轮训练的平均 Loss
    - val_losses: 列表，每一轮验证的平均 Loss
    - train_accs: 列表，每一轮训练的准确率
    - val_accs: 列表，每一轮验证的准确率
    - save_path: 图片保存的路径
    """
    # 获取总共有多少个 epoch
    epochs = range(1, len(train_losses) + 1)

    # 创建一个画布，大小为 12x5 英寸
    plt.figure(figsize=(12, 5))

    # ====================
    # 绘制第一张子图：Loss 曲线
    # ====================
    plt.subplot(1, 2, 1) # 1行2列的第1个位置
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')      # 蓝实线代表训练 Loss
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')  # 红虚线代表验证 Loss
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend() # 显示图例

    # ====================
    # 绘制第二张子图：Accuracy 曲线
    # ====================
    plt.subplot(1, 2, 2) # 1行2列的第2个位置
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')      # 蓝实线代表训练准确率
    plt.plot(epochs, val_accs, 'r--', label='Validation Accuracy')  # 红虚线代表验证准确率
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend() # 显示图例

    # 自动调整子图间距，防止文字重叠
    plt.tight_layout()

    # 保存图片到指定路径
    plt.savefig(save_path)
    print(f"训练曲线已保存至: {save_path}")
    
    # 如果你在能够显示图片的 IDE 中运行，可以把这句取消注释来直接查看
    # plt.show() 
    
    # 释放内存
    plt.close()