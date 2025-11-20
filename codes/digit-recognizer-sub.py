import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 正确的设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)

# 创建 Dataset（无标签）
class TestImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]

# Define model
class NeuralNetwork(nn.Module):     #  继承 nn.Module，所有 PyTorch 的神经网络都应继承自 nn.Module。 这个基类提供了参数管理、前向传播、保存/加载模型等功能。
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()     #将输入张量从形状 (batch_size, 28, 28) 压平为 (batch_size, 784)。因为后续的 Linear 层只能处理一维向量（除了第一个维度是 batch）。
        self.linear_relu_stack = nn.Sequential(         #  是一种顺序容器，会按顺序执行列表中的模块。每一层依次作用于前一层输出。
            nn.Linear(28*28, 512),      # 输入层：784 → 512，全连接层
            nn.ReLU(),      # 激活函数：引入非线性
            nn.Linear(512, 512),        # 隐藏层：512 → 512
            nn.ReLU(),      # 激活函数
            nn.Linear(512, 10)      # 输出层：分类任务，10 类（如 MNIST 数字 0~9，衣服类型等等）
        )

    def forward(self, x):       # 前向传播：forward 函数
        x = self.flatten(x)     # 输入 x 先被压平（flatten），然后送入 linear_relu_stack
        logits = self.linear_relu_stack(x)
        return logits       # 返回的是原始输出（logits），不是概率，因为没有做softmax

model = NeuralNetwork().to(device)      # 创建模型实例并移动到设备

model.load_state_dict(torch.load("../data/digit-recognizer-data/digit-recognizer.pth", weights_only=True))

classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]


test_ori_data = pd.read_csv('../data/digit-recognizer-data/test.csv')

# 转换为tensor
pix_test_tensor = torch.from_numpy(test_ori_data.values).float() / 255.0
pix_test_tensor = pix_test_tensor.reshape(-1, 28, 28)  # 变成 (N, 28, 28)

batch_size = 64

# 转换为Dataset
image_test_dataset = TestImageDataset(pix_test_tensor)

# Create data loaders.
test_dataloader = DataLoader(image_test_dataset, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()     # nn.CrossEntropyLoss()这是一个分类任务中最常用的损失函数。它内部自动对模型输出（logits）做 softmax，然后计算 负对数似然损失（NLL Loss）。


# 推理
print("测试数据形状:", pix_test_tensor.shape)  # 应该是 [28000, 28, 28]
model.eval()
predictions = []
with torch.no_grad():       # 临时关闭自动梯度计算
    for X in test_dataloader:
        X = X.to(device)
        pred = model(X)
        # 把结果从 GPU/MPS 移回 CPU。因为numpy只能在cpu上计算
        pred_labels = pred.argmax(1).cpu().numpy()  # 在第 1 个维度（即类别维度） 上找最大值的索引,输出是一个形状为 (batch_size,) 的 LongTensor，值在 0~9 之间
        predictions.extend(pred_labels)     # 将 pred_labels 中的所有元素逐个添加到 predictions 列表末尾。


# 生成提交文件
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv("../data/digit-recognizer-data/submission.csv", index=False)
print("预测完成，已保存 ../data/digit-recognizer-data/submission.csv")