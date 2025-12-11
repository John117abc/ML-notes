import torch
import torch.nn as nn
from models import LeNet
from utils import get_logger, save_checkpoint,load_config,setup_environment
from data import get_mnist_dataloaders

class Trainer:
    def __init__(self, config):
        self.train_loader = None
        self.val_loader = None
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型、优化器、损失函数
        self.model = LeNet(**config.model.model_args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)
        self.criterion = nn.CrossEntropyLoss()

        # 日志 & 其他工具
        self.logger = get_logger()
        self.best_val_acc = 0.0

    def load_data(self):
        # 或者从外部传入 DataLoader，更灵活
        self.train_loader, self.val_loader = get_mnist_dataloaders(self.config)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        correct = total = 0
        for batch in self.val_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(inputs)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
        acc = 100. * correct / total
        return acc

    def train(self):
        self.load_data()
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_acc = self.validate()

            self.logger.info(f"回合数 {epoch + 1}: 损失={train_loss:.6f}, 准确度={val_acc:.4f}%")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(self.model, self.config.train.save_path)

        self.logger.info(f"训练结束. 最好的准确度是: {self.best_val_acc:.4f}%")
