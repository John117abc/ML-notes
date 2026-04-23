import matplotlib as mpl
mpl.rcParams.update({
    'font.sans-serif': ['Noto Sans CJK JP'],  # 或 WenQuanYi Zen Hei + 避免 \u2212
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix'
})
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from models import LeNet
from utils import get_logger, save_checkpoint, load_config, setup_environment
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

        # 新增：用于记录每轮指标
        self.train_losses = []
        self.val_accuracies = []

    def load_data(self):
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

    def plot_and_save_curves(self):
        """训练结束后绘制并保存 loss 和 accuracy 曲线"""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='训练损失', color='red')
        plt.xlabel('回合')
        plt.ylabel('损失')
        plt.title('训练损失')
        plt.grid(True)
        plt.legend()

        # 验证准确度
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.val_accuracies, label='精度 (%)', color='blue')
        plt.xlabel('回合')
        plt.ylabel('精度 (%)')
        plt.title('精度')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # 保存图像
        save_dir = os.path.dirname(self.config.train.save_path)
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        self.logger.info(f"训练曲线已保存至: {plot_path}")

        # 显示图像（如果环境支持）
        try:
            plt.show()
        except Exception as e:
            self.logger.warning(f"无法显示图像（可能无图形界面）: {e}")

    def train(self):
        self.load_data()

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_acc = self.validate()

            # 保存到列表
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)

            self.logger.info(f"回合数 {epoch + 1}: 损失={train_loss:.6f}, 准确度={val_acc:.4f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(self.model, self.config.train.save_path)

        self.logger.info(f"训练结束. 最佳验证准确度: {self.best_val_acc:.4f}%")

        # 训练完成后绘制曲线
        self.plot_and_save_curves()