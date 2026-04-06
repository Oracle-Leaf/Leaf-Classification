import torch
import numpy as np
import random
import pandas as pd
import os
from train import train_and_evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    
    # 基础参数配置
    base_configs = [
        {"name": "01_Adam_Best", 
         "optimizer_type": "Adam", 
         "lr": 2e-4, "batch_size": 64, 
         "hidden_dims": [1024, 512, 256], 
         "dropout_prob": 0.0, 
         "use_bn": True, 
         "weight_decay": 1e-6},
        {"name": "02_Adam_Robust", 
         "optimizer_type": "Adam", 
         "lr": 1e-4, "batch_size": 50, 
         "hidden_dims": [1024, 512], 
         "dropout_prob": 0.1, 
         "use_bn": True, "weight_decay": 0.0},
        {"name": "03_SGD_Compare", 
         "optimizer_type": "SGD", 
         "lr": 1e-2, "momentum": 0.9, 
         "batch_size": 64, 
         "hidden_dims": [1024, 512, 256], 
         "dropout_prob": 0.0, 
         "use_bn": True, 
         "weight_decay": 1e-6},
        {"name": "04_Adam_Wide", 
         "optimizer_type": "Adam", 
         "lr": 1e-4, 
         "batch_size": 128, 
         "hidden_dims": [1024, 1024, 512], 
         "dropout_prob": 0.2, 
         "use_bn": True, 
         "weight_decay": 0.0}
    ]

    # 要测试的 Epoch 梯度
    epoch_list = [1000]
    
    results_summary = []
    
    print(f"🚀 开始执行全量矩阵实验：4个配置 × {len(epoch_list)}种时长 = {4 * len(epoch_list)}组实验")

    for base in base_configs:
        for epochs in epoch_list:
            # 动态生成实验名称，例如: FINAL_01_Adam_Best_E150
            current_exp_name = f"FINAL_{base['name']}_E{epochs}"
            
            # 准备参数字典
            params = base.copy()
            params.pop("name") # 移除基础名，只留训练参数
            params["num_epochs"] = epochs
            
            print(f"\n▶️ 正在运行: {current_exp_name} (训练{epochs}轮)...")
            
            # 调用训练函数
            try:
                acc = train_and_evaluate(exp_name=current_exp_name, **params)
                results_summary.append({
                    "Config": base['name'],
                    "Epochs": epochs,
                    "Accuracy": acc
                })
            except Exception as e:
                print(f"❌ 实验 {current_exp_name} 失败: {e}")

    # 汇总并保存
    df = pd.DataFrame(results_summary)
    # 按准确率从高到低排序
    df = df.sort_values(by="Accuracy", ascending=False)
    
    print("\n" + "="*60)
    print("📊 最终矩阵实验结果汇总（按表现排序）:")
    print(df.to_string(index=False))
    print("="*60)
    
    df.to_csv("experiment_results/final_matrix_results.csv", index=False)
    print("✅ 结果已保存至 experiment_results/final_matrix_results.csv")

if __name__ == '__main__':
    main()