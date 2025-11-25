import torch
import pandas as pd
from NeuralNetwork import MPLNeuralNetwork,MPLpredict
from torchvision import transforms
from ImageDataset import NoneListLabelDataset
import multiprocessing

# 正确的设备选择
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)

print(f"使用 {device} 设备")

model = MPLNeuralNetwork().to(device)
model.load_state_dict(torch.load("../../data/cifar-10/model.pth", weights_only=True))

# 读取数据
train_label = pd.read_csv('/home/jiangchengxuan/dataset/cifar-10/trainLabels.csv')
# CIFAR-10 的 10 个类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 创建标签到索引的映射
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
# 提取 label 列（第二列），并转换为整数索引
labels_str = train_label['label'].values  # 假设列名为 'label'
labels_index = [class_to_idx[label] for label in labels_str]

# 读取数据
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

cpu_count = multiprocessing.cpu_count()
print(cpu_count)
train_dataset = NoneListLabelDataset(img_dir="/home/jiangchengxuan/dataset/cifar-10/test", transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8192,          # 增大 batch size
    shuffle=False,
    num_workers=cpu_count,           # 启用多进程加载
    pin_memory=True          # 加速 GPU 传输
)


predictions = MPLpredict(train_loader, model, device)  # 预测结果

predictions_name = []
for prediction in predictions:
    predictions_name.append(classes[prediction])


# 生成提交文件
submission = pd.DataFrame({
    "id": range(1, len(predictions) + 1),
    "label": predictions_name
})

submission.to_csv("../../data/cifar-10/submission.csv", index=False)
print("预测完成，已保存 ../data/cifar-10/submission.csv")