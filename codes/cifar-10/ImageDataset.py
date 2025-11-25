import os
import torch as nn
from PIL import Image
from torch.utils.data import Dataset

# 创建 Dataset（无标签）
class NoneListLabelDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (str): 图片所在文件夹路径
            transform (callable, optional): 可选的图像变换
        """
        self.img_dir = img_dir
        self.transform = transform

        self.img_names = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 按文件名中的数字排序
        self.img_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        # 将标签转换为Tensor类型
        if self.transform:
            image = self.transform(image)

        return image

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

        self.img_names = [f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # 按文件名中的数字排序
        self.img_names.sort(key=lambda x: int(os.path.splitext(x)[0]))

        assert len(self.img_names) == len(self.labels), \
            f"Number of images ({len(self.img_names)}) != number of labels ({len(self.labels)})"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        # 将标签转换为Tensor类型

        label = nn.tensor(label, dtype=nn.long).squeeze()

        if self.transform:
            image = self.transform(image)

        return image, label