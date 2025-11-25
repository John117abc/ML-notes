import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from NeuralNetwork import MPLNeuralNetwork
from ImageDataset import ListLabelDataset
from NNTrain import MPLtrain
# 尝试不适用CNN，使用全链接MPL进行图像分类训练

# 读取数据
train_label = pd.read_csv('/home/jiangchengxuan/dataset/cifar-10/trainLabels.csv')
# CIFAR-10 的 10 个类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 创建标签到索引的映射
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# 加载原始像素
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 提取 label 列（第二列），并转换为整数索引
labels_str = train_label['label'].values  # 假设列名为 'label'
labels_index = [class_to_idx[label] for label in labels_str]

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

print(mean)
print(std)

# 再读取
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

model = MPLNeuralNetwork().to(device)      # 创建模型实例并移动到设备
print(model)

loss_fn = nn.CrossEntropyLoss()     # nn.CrossEntropyLoss()这是一个分类任务中最常用的损失函数。它内部自动对模型输出（logits）做 softmax，然后计算 负对数似然损失（NLL Loss）。
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)   # 使用随机梯度下降（Stochastic Gradient Descent） 优化器。model.parameters()：告诉优化器“哪些参数需要更新”——即模型中所有可学习的权重和偏置。lr=1e-3：学习率（learning rate），控制每次参数更新的步长。

epochs = 10
for t in range(epochs):     # 类似于java的 for(int i = 0;i<10;i++)
    print(f"回合 {t+1}\n-------------------------------")
    MPLtrain(train_loader, model, loss_fn, optimizer,device)
print("完成!")

torch.save(model.state_dict(), "../data/cifar-10/model.pth")
print("训练模型保存至：../data/cifar-10/model.pth")