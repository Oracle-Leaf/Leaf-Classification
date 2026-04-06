import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# 直接导入你现有的类，代码极致清爽！
from dataset import LeavesDataset
from model import FlexibleMLP

def main():
    # ==========================================
    # 1. 绝对路径配置 (已修复 final_results 拼写)
    # ==========================================
    MODEL_PATH = r"C:\Users\HP\Desktop\Data_Science_Practice_Assignment\Assignment_1\experiment_results\FINAL_03_SGD_Compare_E200\best_model.pth"
    TEST_CSV_PATH = r"C:\Users\HP\Desktop\Data_Science_Practice_Assignment\Assignment_1\dataset\test.csv"
    DATA_ROOT = r"C:\Users\HP\Desktop\Data_Science_Practice_Assignment\Assignment_1\dataset"
    OUTPUT_DIR = r"C:\Users\HP\Desktop\Data_Science_Practice_Assignment\Assignment_1\final_results"
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")

    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # ==========================================
    # 2. 数据预处理 (完全对齐训练时的验证集)
    # ==========================================
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 3. 初始化模型并加载权重
    # 对应 03_SGD_Compare 的配置
    # ==========================================
    model = FlexibleMLP(
        input_dim=224*224*3, 
        hidden_dims=[1024, 512, 256], 
        num_classes=176, 
        use_bn=True, 
        dropout_prob=0.0
    )
    print(f"📦 加载权重:\n{MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # 锁定 BatchNorm

    # ==========================================
    # 4. 阶段一：计算准确率和错误分析 
    # (基于 train.csv 划分出的带标签测试集)
    # ==========================================
    print("\n" + "="*50)
    print("📊 阶段一：计算期末准确率与错误分析 (写报告用)")
    print("="*50)
    
    test_dataset = LeavesDataset(DATA_ROOT, mode='test', transforms=transforms_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    idx_to_class = test_dataset.idx_to_class # 获取数字到英文名字的字典

    all_preds = []
    all_labels = []
    error_cases = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Accuracy"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            pred_list = predicted.cpu().numpy()
            label_list = labels.cpu().numpy()
            
            all_preds.extend(pred_list)
            all_labels.extend(label_list)

            # 收集错误样本用于 Error Analysis
            for i in range(len(label_list)):
                if pred_list[i] != label_list[i]:
                    true_name = idx_to_class[label_list[i]]
                    pred_name = idx_to_class[pred_list[i]]
                    error_cases.append(f"真实: [{true_name}] -> 错认为: [{pred_name}]")

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n🏆 Test Accuracy (准确率): {test_acc * 100:.2f}%")
    print(f"🏆 Test Macro-F1: {test_f1 * 100:.2f}%")

    print("\n🚨 Top 5 最易混淆的树叶 (放入报告的 Error Analysis):")
    top_errors = Counter(error_cases).most_common(5)
    for error_pair, count in top_errors:
        print(f"混淆 {count} 次 | {error_pair}")

    # ==========================================
    # 5. 阶段二：为 test.csv 生成预测结果
    # ==========================================
    print("\n" + "="*50)
    print("📄 阶段二：生成 submission.csv")
    print("="*50)
    
    df_test = pd.read_csv(TEST_CSV_PATH)
    predictions = []
    
    with torch.no_grad():
        for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Predicting CSV"):
            img_path = os.path.join(DATA_ROOT, row['image'])
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transforms_test(image).unsqueeze(0).to(device)
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # 直接将预测的数字翻译成对应的英文树叶名字
                pred_label_str = idx_to_class[predicted.item()]
                predictions.append(pred_label_str)
            except Exception as e:
                print(f"读取图片失败: {img_path}, 错误: {e}")
                predictions.append("unknown")

    df_test['label'] = predictions
    df_test.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 预测完成！结果 CSV 已保存至:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()