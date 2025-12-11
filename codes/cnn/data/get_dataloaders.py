import torch
from .dataset import CsvMNISTDataset
from torch.utils.data import Dataset, DataLoader, random_split

def get_mnist_dataloaders(config):
    full_dataset = CsvMNISTDataset(config.data.train_csv, has_labels=True)

    # 拆分
    train_size = int((1 - config.data.val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
    return train_loader, val_loader