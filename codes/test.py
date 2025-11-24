import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from PIL import Image


# 读取数据
train_label = pd.read_csv('/home/jiangchengxuan/dataset/cifar-10/trainLabels.csv')

class ListLabelDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        """
        Args:
            img_dir (str): 图片所在文件夹路径
            labels (list): 标签列表，labels[i] 对应第 i 张图的类别
            transform (callable, optional): 可选的图像变换
        """
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

        # 获取所有图片文件名，并排序以保证顺序一致
        self.img_names = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_names.sort()  # 确保顺序固定！非常重要

        # 安全检查：数量是否匹配
        assert len(self.img_names) == len(self.labels), \
            f"Number of images ({len(self.img_names)}) != number of labels ({len(self.labels)})"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        # 将标签转换为Tensor类型

        label = torch.tensor(label, dtype=torch.long).squeeze()

        if self.transform:
            image = self.transform(image)

        return image, label

# 加载原始像素
transform = transforms.Compose([
    transforms.ToTensor(),
])

labels_index = train_label.iloc[:, 0:-1].values.astype(np.int32).tolist()


dataset = ListLabelDataset(img_dir="/home/jiangchengxuan/dataset/cifar-10/train",labels = labels_index ,transform=transform)
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

# 初始化统计量
mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0

# 遍历所有 batch
for images, _ in dataloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)  # (批次大小, 通道数, 高*宽)
    mean += images.mean(dim=2).sum(dim=0)
    std += images.std(dim=2).sum(dim=0)
    nb_samples += batch_samples

# 计算全局均值和标准差
mean /= nb_samples
std /= nb_samples


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])

train_dataset = ListLabelDataset(img_dir="/home/jiangchengxuan/dataset/cifar-10/train",labels = labels_index, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 正确的设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)

print(f"使用 {device} 设备")

# Define model
class NeuralNetwork(nn.Module):     #  继承 nn.Module，所有 PyTorch 的神经网络都应继承自 nn.Module。 这个基类提供了参数管理、前向传播、保存/加载模型等功能。
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

model = NeuralNetwork().to(device)      # 创建模型实例并移动到设备
print(model)

loss_fn = nn.CrossEntropyLoss()     # nn.CrossEntropyLoss()这是一个分类任务中最常用的损失函数。它内部自动对模型输出（logits）做 softmax，然后计算 负对数似然损失（NLL Loss）。
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)   # 使用随机梯度下降（Stochastic Gradient Descent） 优化器。model.parameters()：告诉优化器“哪些参数需要更新”——即模型中所有可学习的权重和偏置。lr=1e-3：学习率（learning rate），控制每次参数更新的步长。


def train(dataloader, model, loss_fn, optimizer):
    """
    :param dataloader: 提供批量数据（如 DataLoader 实例）
    :param model: 要训练的神经网络；
    :param loss_fn: 损失函数
    :param optimizer: 优化器。
    :return:
    """
    size = len(dataloader.dataset)      # 获取整个训练集样本总数（用于打印进度）。
    model.train()       #  将模型切换到 训练模式（training mode）。 影响某些层的行为，例如： Dropout：训练时启用，推理时关闭； BatchNorm：训练时用 batch 统计量，推理时用全局统计量。 如果不调用 .train()，模型可能表现异常（尤其用了这些层时）。
    for batch, (X, y) in enumerate(dataloader):   #  每次返回一个 batch 的 (X, y)，X：输入图像。y：标签
        # 调试：打印类型
        if batch == 0:
            print("X.shape:", X.shape)  # 应为 [B, 3, 32, 32]
            print("y.shape:", y.shape)  # 应为 [B]，不是 [B, 1]！
            print("y.dtype:", y.dtype)  # 应为 torch.int64 (long)
        X, y = X.to(device), y.to(device)

        pred = model(X)     # 调用模型的 forward() 方法，得到 logits（shape [64, 10]），向前传播
        loss = loss_fn(pred, y)     # 计算当前 batch 的平均损失（标量）。

        # 反向传播更新参数
        loss.backward()     # 自动计算损失对所有可学习参数的梯度（通过反向传播算法）。 梯度会累加到每个参数的 .grad 属性中。
        optimizer.step()        # 根据当前梯度（.grad）和优化器规则（如 SGD: w = w - lr * grad）更新参数。
        optimizer.zero_grad()       #  清空梯度缓冲区（将所有 .grad 设为 None 或 0）。 ⚠️ 如果不清零，梯度会累积！ 下一次 backward() 会把新梯度加到旧梯度上，导致错误更新。
        if batch % 100 == 0:        # 打印训练进度
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


epochs = 10
for t in range(epochs):     # 类似于java的 for(int i = 0;i<10;i++)
    print(f"回合 {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
print("完成!")

torch.save(model.state_dict(), "../data/cifar-10/model.pth")
print("训练模型保存至：../data/cifar-10/model.pth")