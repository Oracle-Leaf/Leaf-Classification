import torch
# 导入 PyTorch 的神经网络模块，所有的网络层（如全连接层、激活函数等）都在这里面
import torch.nn as nn 

class FlexibleMLP(nn.Module):
    """
    自定义的灵活 MLP 模型。
    继承自 nn.Module，这是 PyTorch 中所有神经网络模型的基类。
    """
    
    # 初始化函数，定义了模型的各种组件和超参数开关
    def __init__(self, input_dim=224*224*3, hidden_dims=[512, 256], num_classes=176, use_bn=False, dropout_prob=0.0):
        """
        参数说明 (这里也是你做实验时可以修改的开关):
        - input_dim: 输入特征的维度。图像大小为 224x224，RGB 三通道，所以展平后是 224*224*3 = 150528
        
        ========== 实验参数开关提示 ==========
        - hidden_dims: 隐藏层维度列表。决定了有几个隐藏层，以及每层有多少个神经元。
            # 尝试 1: 单隐藏层，较宽 -> hidden_dims=[1024]
            # 尝试 2: 双隐藏层 (默认) -> hidden_dims=[512, 256]
            # 尝试 3: 深层网络 -> hidden_dims=[512, 256, 128, 64]
        
        - use_bn: 是否使用批量归一化 (Batch Normalization)。用于加速收敛，稳定训练。
            # 尝试 1: 不使用 BN -> use_bn=False
            # 尝试 2: 使用 BN -> use_bn=True
            
        - dropout_prob: Dropout 概率，用于随机丢弃神经元，防止过拟合。
            # 尝试 1: 不使用 Dropout -> dropout_prob=0.0
            # 尝试 2: 轻度 Dropout -> dropout_prob=0.2
            # 尝试 3: 中度 Dropout -> dropout_prob=0.5
        =======================================
        """
        # 调用父类 nn.Module 的初始化方法，这是必须的步骤
        super(FlexibleMLP, self).__init__()
        
        # 将输入的 2D 图像特征图展平为 1D 向量
        # 例如将形状为 (Batch_Size, 3, 224, 224) 的张量展平为 (Batch_Size, 150528)
        self.flatten = nn.Flatten()
        
        # 创建一个空列表，用来按顺序存放我们的网络层
        layers = []
        
        # 当前层的输入维度，第一层就是图像展平后的维度
        current_dim = input_dim
        
        # 遍历 hidden_dims 列表，动态构建隐藏层
        for hidden_dim in hidden_dims:
            # 添加一个全连接层 (Linear Layer)，输入维度是 current_dim，输出维度是 hidden_dim
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # 如果开启了 Batch Normalization，就在全连接层之后加上 BN 层
            if use_bn:
                # nn.BatchNorm1d 用于处理 1D 的向量数据
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 添加 ReLU 激活函数，增加模型的非线性表达能力
            layers.append(nn.ReLU())
            
            # 如果 dropout 概率大于 0，则添加 Dropout 层
            if dropout_prob > 0.0:
                # 按照 dropout_prob 的概率随机将部分神经元的输出置为 0
                layers.append(nn.Dropout(p=dropout_prob))
                
            # 更新当前层维度，作为下一层的输入维度
            current_dim = hidden_dim
            
        # 隐藏层构建完毕后，添加最后一层：输出层
        # 输出层的输入是最后一个隐藏层的输出，输出维度是类别数 (num_classes)
        layers.append(nn.Linear(current_dim, num_classes))
        
        # 将列表中的层打包成一个整体模块 nn.Sequential
        # 数据在传入 self.network 时，会按列表里的顺序依次通过每一层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。定义了数据 x 在网络中的流向。
        当你在代码中调用 model(x) 时，实际上就是在运行这个函数。
        """
        # 1. 首先把输入的图像 x 展平成一维向量
        x = self.flatten(x)
        # 2. 将展平后的向量送入我们刚刚构建的网络序列中计算
        logits = self.network(x)
        
        # 返回网络的原始输出 (称为 logits)。
        # 注意：这里我们没有加 Softmax，因为 PyTorch 的交叉熵损失函数 (CrossEntropyLoss) 内部会自动处理。
        return logits

# 这是一个简单的测试代码，当你直接运行这个脚本时会执行，用来验证模型能不能跑通
if __name__ == '__main__':
    # 模拟一个 Batch 的数据，假设 batch_size=4，3 个通道，高宽都是 224
    dummy_input = torch.randn(4, 3, 224, 224) 
    
    # 实例化我们的模型，这里使用默认参数
    model = FlexibleMLP()
    
    # 让模拟数据通过模型，得到输出
    output = model(dummy_input)
    
    # 打印输出张量的形状。期望输出是 [4, 176]，即 4 个样本，每个样本 176 个类别的得分
    print(f"模型输出的形状为: {output.shape}")