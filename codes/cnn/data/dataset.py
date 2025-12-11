import pandas as pd
import torch
from torch.utils.data import Dataset

class CsvMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None, has_labels=True):
        """
        Args:
            csv_file (str): CSV 文件路径。
            transform (callable, optional): 可选的图像变换（如 ToTensor, Normalize）。
            has_labels (bool): CSV 是否包含标签列（训练集有，测试集可能没有）。
        """
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        self.transform = transform

        if has_labels:
            # 第一列是 label，后面是像素
            self.labels = self.data.iloc[:, 0].values
            self.pixels = self.data.iloc[:, 1:].values
        else:
            # 没有标签，测试集
            self.labels = None
            self.pixels = self.data.values

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        # 获取第 idx 个样本的像素（784 维）
        pixel_array = self.pixels[idx].astype('float32')

        # 重塑为 28x28，并添加通道维度 -> (1, 28, 28)
        image = pixel_array.reshape(28, 28)  # 灰度图
        image = image / 255.0  # 归一化到 [0, 1]

        # 转为 PyTorch 张量 (H, W) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)

        # 应用额外的 transform（如 Normalize）
        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = int(self.labels[idx])
            return image, label
        else:
            return (image,)