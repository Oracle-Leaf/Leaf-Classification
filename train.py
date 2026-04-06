import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from dataset import LeavesDataset
from model import FlexibleMLP
from utils import set_seed, plot_metrics

# 【修改点 1】把需要控制的变量作为函数的参数传进来！
def train_and_evaluate(
    exp_name="baseline",     # 实验名称，用于自动建文件夹
    lr=1e-3,                 # 学习率
    batch_size=128,          # 批大小
    hidden_dims=[512, 256],  # 隐藏层结构
    use_bn=True,             # 是否使用 BN
    dropout_prob=0.2,        # Dropout 概率
    weight_decay=1e-4,       # 权重衰减
    num_epochs=20,           # 训练轮数
    optimizer_type="Adam", 
    momentum=0.9
):
    print(f"\n{'='*50}\n正在运行实验: {exp_name}\n{'='*50}")
    
    # 【修改点 2】自动创建实验专属文件夹，保证模型和图片不乱
    save_dir = os.path.join('experiment_results', exp_name)
    os.makedirs(save_dir, exist_ok=True) # 如果文件夹不存在就自动创建
    
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前使用的计算设备是: {device}")
    
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),       
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),               
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = 'dataset' 
    train_dataset = LeavesDataset(data_root, mode='train', transforms=transforms_train)
    val_dataset = LeavesDataset(data_root, mode='valid', transforms=transforms_valid)

    # 使用传入的 batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 使用传入的模型结构参数
    model = FlexibleMLP(hidden_dims=hidden_dims, use_bn=use_bn, dropout_prob=dropout_prob) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 使用传入的 lr 和 weight_decay
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("不支持的优化器类型！")
    best_val_acc = 0.0
    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []

    for epoch in range(num_epochs):
        # ---------------- 训练阶段 ----------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
        
        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_train_acc.append(epoch_train_acc)
        history_val_acc.append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # 【修改点 3】将最佳模型保存到专属文件夹中
            model_save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_save_path)

        # ==============================================================
        # 🌟 救命护身符：把这段画图代码放进 epoch 循环里面！（注意缩进）
        # 让它每逢 10 的倍数，或者最后一轮的时候，立刻把现有的数据画成图保存！
        # ==============================================================
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            plot_save_path = os.path.join(save_dir, 'training_curves.png')
            plot_metrics(history_train_loss, history_val_loss, history_train_acc, history_val_acc, save_path=plot_save_path)
            # 因为你的 utils.py 里已经有 plt.close() 了，所以这里直接调函数非常安全！

    print(f"[{exp_name}] 训练结束！最佳验证集准确率为: {best_val_acc:.4f}")
    
    # 【修改点 4】将训练曲线保存到专属文件夹中
    plot_save_path = os.path.join(save_dir, 'training_curves.png')
    plot_metrics(history_train_loss, history_val_loss, history_train_acc, history_val_acc, save_path=plot_save_path)
    
    return best_val_acc # 返回最高准确率，方便后续批量收集结果

# ==========================================
# 单参数快速测试入口 (只保留这一个即可)
# ==========================================
if __name__ == '__main__':
    print("开始单参数快速测试...")
    train_and_evaluate(
        exp_name="test_single_run",  # 测试用的文件夹名称
        lr=1e-3,
        batch_size=128,
        hidden_dims=[512, 256],
        use_bn=True,
        dropout_prob=0.2,
        num_epochs=2  # 【关键】为了快速验证代码有没有报错，我们只跑 2 轮！
    )