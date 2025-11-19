import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class MyImageDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data  # ['img1.jpg', 'img2.jpg', ...]
        self.labels = labels            # [0, 1, 2, ...]
    def __len__(self):
        return len(self.image_data)
    def __getitem__(self, idx):
        image = self.image_data[idx]
        label = self.labels[idx].item()  # 如果 labels 是 (N, 1) 形状的张量，则 .item() 可以提取标量值
        return image, label

SEED = 20

train_ori_data = pd.read_csv('../data/digit-recognizer-data/train.csv')

all_features = train_ori_data.iloc[:,1:]

all_label = train_ori_data.iloc[:,0:1]

pix_train, pix_test, number_train, number_test = train_test_split(all_features,all_label,test_size=0.2, random_state=SEED)

# 转换为tensor
pix_train_tensor = torch.from_numpy(pix_train.values).float()
pix_test_tensor = torch.from_numpy(pix_test.values).float()
number_train_tensor = torch.from_numpy(number_train.values)
number_test_tensor = torch.from_numpy(number_test.values)

batch_size = 64

# 转换为Dataset
image_train_dataset = MyImageDataset(pix_train_tensor, number_train_tensor)
image_test_dataset = MyImageDataset(pix_test_tensor,number_test_tensor)

# Create data loaders.
train_dataloader = DataLoader(image_train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(image_test_dataset, batch_size=batch_size)


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
    model.train()       # 非常重要！ 将模型切换到 训练模式（training mode）。 影响某些层的行为，例如： Dropout：训练时启用，推理时关闭； BatchNorm：训练时用 batch 统计量，推理时用全局统计量。 如果不调用 .train()，模型可能表现异常（尤其用了这些层时）。
    for batch, (X, y) in enumerate(dataloader):   #  每次返回一个 batch 的 (X, y)，X：输入图像。y：标签
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

def test(dataloader, model, loss_fn):
    """
    在测试集上评估训练好的模型； 计算 平均损失（Avg loss） 和 分类准确率（Accuracy）； 不进行反向传播或参数更新（因为只是评估，不是训练）； 使用 torch.no_grad() 提升效率并节省内存。
    :param dataloader:  测试数据加载器（如 DataLoader(test_dataset)）；
    :param model:   已训练的模型；
    :param loss_fn: 与训练时相同的损失函数（如 CrossEntropyLoss）。
    """
    size = len(dataloader.dataset)      # 总样本数，如 10000（MNIST 测试集）
    num_batches = len(dataloader)       # 总 batch 数，如 157（10000 / 64 ≈ 157）,这两个值用于后续计算平均损失和整体准确率。
    model.eval()        # 将模型设置为 评估模式（evaluation mode）。对应训练时的 model.train()，两者必须成对使用！
    test_loss, correct = 0, 0       # 累计所有 batch 的损失总和；    累计预测正确的样本总数。
    with torch.no_grad():       # with类似与java中的try catch finally，禁用梯度计算：with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)     # 前向传播，得到预测 logits
            test_loss += loss_fn(pred, y).item()        # loss_fn(pred, y)：计算当前 batch 的平均损失（标量张量）；
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # pred.argmax(1)在对所有类别预测中找到数值最大的--预测最像的，如果和标签y相等证明预测对了
    test_loss /= num_batches        #   平均每个 batch 的损失（注意：不是每个样本！但因为 CrossEntropyLoss 默认对 batch 求平均，所以等价于平均每个样本的 loss）；
    correct /= size     # 整体准确率（正确样本数 / 总样本数）。
    print(f"失败: \n 精确度: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):     # 类似于java的 for(int i = 0;i<10;i++)
    print(f"回合 {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("完成!")