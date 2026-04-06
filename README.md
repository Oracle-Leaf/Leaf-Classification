# 🍁 Pure MLP for Fine-Grained Image Classification: An Empirical Study

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)

## 📌 Project Overview
This repository contains a rigorous, empirical study on training a **pure Multi-Layer Perceptron (MLP)** for a challenging 176-category fine-grained image classification task (Leaves Dataset). 

To deeply understand the fundamental mechanics of neural networks, **this project strictly avoids Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs)**. Instead, it relies entirely on flattened high-dimensional inputs ($224 \times 224 \times 3 = 150,528$ dimensions), forcing the pure MLP to learn without spatial inductive biases (like local receptive fields or translation invariance).

Through systematic hyperparameter tuning and extensive logging, this project achieved a **~55% test accuracy**, a highly competitive result for a pure MLP on this complex dataset.

You can download needed dataset via https://pan.baidu.com/s/19QFv1cK1WocOxe9M5vXe0g?pwd=xatm 

## 🚀 Key Highlights for Resume / Portfolios
* **Systematic Ablation Studies:** Conducted strictly controlled experiments across 6 dimensions: Learning Rate, Batch Size, Network Depth vs. Width, Dropout, Weight Decay, and Batch Normalization.
* **Deep Error Analysis:** Investigated the failure modes of MLPs on spatial data, proving empirically why the destruction of 2D topology leads to confusion among structurally similar classes (e.g., needle-leaf species).
* **Optimization vs. Generalization:** Compared Adam and SGD over 200-epoch runs, observing the classic trade-off where Adam converges faster, but SGD avoids sharp local minima to achieve superior ultimate generalization.
* **Extreme Overfitting Analysis:** Extended training to 1,000 epochs to observe and document the absence of textbook "U-curve" validation divergence, attributing it to 1D topology destruction and BN regularization.

## 📊 Final Performance
The final selected configuration (`SGD-Compare`) achieved the following metrics on the held-out test set:
* **Test Accuracy:** 54.62%
* **Macro-F1 Score:** 52.61%
*(The proximity between Accuracy and Macro-F1 confirms robust feature learning across a long-tailed, 176-class distribution, rather than simply overfitting to majority classes.)*

## 💡 Key Empirical Insights
1. **Batch Normalization is Non-Negotiable:** Disabling BN caused accuracy to plummet to ~4.7% (random guessing). BN is critical to mitigating internal covariate shift when handling 150,000+ length vectors.
2. **Capacity Over Regularization:** Heavy Dropout (0.6) or strong Weight Decay ($1e-2$) caused severe underfitting. Since the MLP inherently struggles with complex spatial patterns, it heavily relies on its full parameter capacity early on.
3. **Width Beats Depth:** A wider, shallower network (`1024 -> 512`) outperformed deeper configurations (`512 -> 256 -> 128 -> 64`), circumventing gradient vanishing without the need for residual connections.

## 📂 Repository Structure
```text
.
├── dataset.py            # Custom PyTorch Dataset with data augmentation
├── model.py              # Highly flexible MLP class with toggles for depth, BN, and Dropout
├── train.py              # Core training loop with modular evaluation and early-stopping metrics
├── utils.py              # Global seed fixing for 100% reproducibility & plotting tools
├── run_experiments.py    # Automation script for the 20-epoch hyperparameter ablation grid
├── run_final.py          # Script for extended 200-1000 epoch deep-dive training
├── report.pdf            # Comprehensive 2-page LaTeX academic report of all findings
└── experiment_results/   # Directory containing generated loss/accuracy curves and saved weights
