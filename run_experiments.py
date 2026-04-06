from train import train_and_evaluate
import pandas as pd

def main():
    print("🚀 开始执行精细化全自动调参实验！")
    
    results_summary = []
    EPOCHS = 50  

    # ==========================================
    # 0. 定义基准配置 (Baseline)
    # 所有其他实验都将基于这个配置进行微调
    # ==========================================
    baseline = {
        "lr": 1e-3, 
        "batch_size": 128, 
        "hidden_dims": [512, 256], 
        "dropout_prob": 0.2, 
        "use_bn": True, 
        "weight_decay": 1e-4, 
        "num_epochs": EPOCHS
    }

    # 建立一个实验清单，用 **baseline 继承基准配置，并覆盖我们要修改的变量
    experiments = [
        # --- 0. 基准对照组 ---
        {"exp_name": "00_Baseline", **baseline},

        # --- 1. 学习率 (Learning Rate) 精细搜索 ---
        # 探究：寻找最佳下降步长。1e-4 太慢，1e-2 震荡，中间哪个值最好？
        {"exp_name": "LR_01_1e-4", **baseline, "lr": 1e-4},
        {"exp_name": "LR_02_5e-4", **baseline, "lr": 5e-4},
        # 1e-3 就是 Baseline
        {"exp_name": "LR_03_5e-3", **baseline, "lr": 5e-3},
        {"exp_name": "LR_04_1e-2", **baseline, "lr": 1e-2},

        # --- 2. 批大小 (Batch Size) 精细搜索 ---
        # 探究：内存允许的情况下，Batch Size 变化对泛化的影响
        {"exp_name": "BS_01_32", **baseline, "batch_size": 32},
        {"exp_name": "BS_02_64", **baseline, "batch_size": 64},
        # 128 就是 Baseline
        {"exp_name": "BS_03_256", **baseline, "batch_size": 256},

        # --- 3. 模型容量 (Hidden Dims) 精细搜索 ---
        # 探究：从单层到四层，宽度从 1024 到 128，模型的表达能力极限在哪？
        {"exp_name": "NET_01_wide_1024", **baseline, "hidden_dims": [1024]},
        {"exp_name": "NET_02_1024_512", **baseline, "hidden_dims": [1024, 512]},
        # [512, 256] 就是 Baseline
        {"exp_name": "NET_03_512_256_128", **baseline, "hidden_dims": [512, 256, 128]},
        {"exp_name": "NET_04_512_256_128_64", **baseline, "hidden_dims": [512, 256, 128, 64]},

        # --- 4. Dropout 概率精细搜索 ---
        # 探究：什么程度的随机失活能最好地防止过拟合？
        {"exp_name": "DROP_01_0.0", **baseline, "dropout_prob": 0.0},
        {"exp_name": "DROP_02_0.1", **baseline, "dropout_prob": 0.1},
        # 0.2 就是 Baseline
        {"exp_name": "DROP_03_0.4", **baseline, "dropout_prob": 0.4},
        {"exp_name": "DROP_04_0.6", **baseline, "dropout_prob": 0.6},

        # --- 5. L2 正则化 (Weight Decay) 精细搜索 ---
        # 探究：惩罚权重到底需要多大力度？
        {"exp_name": "WD_01_0.0", **baseline, "weight_decay": 0.0},
        {"exp_name": "WD_02_1e-5", **baseline, "weight_decay": 1e-5},
        # 1e-4 就是 Baseline
        {"exp_name": "WD_03_1e-3", **baseline, "weight_decay": 1e-3},
        {"exp_name": "WD_04_1e-2", **baseline, "weight_decay": 1e-2},

        # --- 6. 批量归一化 (Batch Norm) 开关 ---
        {"exp_name": "BN_01_False", **baseline, "use_bn": False},
    ]

    # ==========================================
    # 依次执行所有配置
    # ==========================================
    for i, config in enumerate(experiments):
        exp_name = config.pop("exp_name") # 取出实验名，剩下的都是传给函数的参数
        print(f"\n[{i+1}/{len(experiments)}] 正在准备执行实验: {exp_name} ...")
        
        # 动态传入所有参数
        acc = train_and_evaluate(exp_name=exp_name, **config)
        results_summary.append({"Experiment": exp_name, "Accuracy": acc})

    # ==========================================
    # 汇总并保存
    # ==========================================
    print("\n" + "="*50)
    print("🎉 所有精细化实验运行完毕！最终结果汇总：")
    
    df_results = pd.DataFrame(results_summary)
    print(df_results.to_string(index=False))
    
    df_results.to_csv("experiment_results/final_results_fine_grained.csv", index=False)
    print("="*50)
    print("汇总成绩单已保存至: experiment_results/final_results_fine_grained.csv")

if __name__ == '__main__':
    main()