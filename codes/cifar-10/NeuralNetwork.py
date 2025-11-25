import torch
from torch import nn

# 定义模型 model
class MPLNeuralNetwork(nn.Module):     #  继承 nn.Module，所有 PyTorch 的神经网络都应继承自 nn.Module。 这个基类提供了参数管理、前向传播、保存/加载模型等功能。
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()     #将输入张量从形状 (batch_size, 28, 28) 压平为 (batch_size, 784)。因为后续的 Linear 层只能处理一维向量（除了第一个维度是 batch）。
        self.linear_relu_stack = nn.Sequential(         #  是一种顺序容器，会按顺序执行列表中的模块。每一层依次作用于前一层输出。
            nn.Linear(32*32* 3, 512),      # 输入层：784 → 512，全连接层
            nn.ReLU(),      # 激活函数：引入非线性
            nn.Linear(512, 512),        # 隐藏层：512 → 512
            nn.ReLU(),      # 激活函数
            nn.Linear(512, 10)      # 输出层：分类任务，10 类（如 MNIST 数字 0~9，衣服类型等等）
        )

    def forward(self, x):       # 前向传播：forward 函数
        x = self.flatten(x)     # 输入 x 先被压平（flatten），然后送入 linear_relu_stack
        logits = self.linear_relu_stack(x)
        return logits       # 返回的是原始输出（logits），不是概率，因为没有做softmax

# 预测
def MPLpredict(dataloader, model, device):
    """
    在无标签测试集上进行预测，并打印进度。

    Args:
        dataloader: 测试数据加载器（每个 batch 只包含图像）
        model: 已训练的模型
        device: 设备（'cuda' 或 'cpu'）

    Returns:
        list: 所有预测的类别索引（整数列表）
    """
    model.eval()
    predictions = []
    total_samples = len(dataloader.dataset)
    processed = 0

    with torch.no_grad():
        for batch_idx, X in enumerate(dataloader):
            X = X.to(device)
            pred = model(X)
            pred_labels = pred.argmax(1).cpu().tolist()
            predictions.extend(pred_labels)

            # 更新已处理样本数
            processed += X.size(0)

            # 每处理一定数量或最后一个 batch 时打印进度
            if (batch_idx + 1) % 50 == 0 or processed == total_samples:
                print(f"已处理: [{processed}/{total_samples}] ({100.0 * processed / total_samples:.1f}%)")

    return predictions