# MNIST Handwritten Digit Classification with CNN

使用 PyTorch 实现卷积神经网络（CNN），在 MNIST 手写数字数据集上完成 10 分类任务。

## 结果

- **测试集准确率：99.02%**
- 最佳 epoch：Epoch 4（99.04%）
- 训练环境：WSL2 + Ubuntu 24.04 + Python 3.12 + PyTorch 2.14.1 (CUDA 13.0)

## 环境

- WSL2 + Ubuntu 24.04
- Python 3.12.14
- PyTorch 2.14.1+cu130
- torchvision, matplotlib, numpy, jupyter
- 使用 `uv` 管理 Python 环境

## 项目结构

```text
week3-mnist/
├── data/                    # MNIST 数据（自动下载）
├── week3_mnist.ipynb        # 主 notebook
├── mnist_cnn.pth            # 训练好的模型
├── training_curves.png      # 训练曲线
├── confusion_matrix.png     # 混淆矩阵
├── misclassified.png        # 错分样本
├── requirements.txt
└── README.md
```

## 运行方式

```bash
# 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook
```

打开 `week3_mnist.ipynb`，从第一个 cell 依次运行。

## 模型结构

```text
Input (1, 28, 28)
  ↓ Conv2d(1, 32, 3×3) + ReLU
  ↓ Conv2d(32, 64, 3×3) + ReLU
  ↓ MaxPool2d(2)          → (64, 12, 12)
  ↓ Dropout(0.25)
  ↓ Flatten               → 9216
  ↓ Linear(9216, 128) + ReLU
  ↓ Dropout(0.5)
  ↓ Linear(128, 10)
  ↓ log_softmax
```

## 实验结果

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1 | 0.1839 | 0.9451 | 0.0477 | 0.9842 |
| 2 | 0.0848 | 0.9748 | 0.0387 | 0.9883 |
| 3 | 0.0641 | 0.9814 | 0.0347 | 0.9893 |
| 4 | 0.0525 | 0.9837 | 0.0310 | **0.9904** |
| 5 | 0.0442 | 0.9861 | 0.0350 | 0.9902 |

## 每类准确率

| 数字 | 准确率 | 数字 | 准确率 |
|---|---|---|---|
| 0 | 99.39% | 5 | 99.55% |
| 1 | 99.82% | 6 | **96.97%** |
| 2 | 99.03% | 7 | 99.12% |
| 3 | 99.60% | 8 | 99.08% |
| 4 | 99.80% | 9 | 97.72% |

## 结果分析

- 模型对数字 **1（99.82%）、4（99.80%）、3（99.60%）** 识别最好，这些数字笔画简单、结构独特。
- 数字 **6（96.97%）、9（97.72%）** 识别率相对较低。
- 原因：6 和 9 形状相似（圈 + 尾巴方向相反），手写变体多，部分样本即使人眼也难以辨认。
- 训练到 Epoch 5 时，Test Loss 回升，出现轻微过拟合迹象，Epoch 4 是最佳状态。
- 共有 98 个错分样本（占测试集 0.98%）。

## 改进方向

- 加入学习率衰减（StepLR / CosineAnnealing）
- Early Stopping，避免过拟合
- 数据增强（随机旋转、平移）
- 尝试 Batch Normalization
- 尝试更深或更宽的网络

## 参考

- [PyTorch MNIST 官方示例](https://github.com/pytorch/examples/tree/main/mnist)
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)