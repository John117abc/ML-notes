import torch
import pandas as pd
from models import LeNet
from utils import get_logger,load_config,setup_environment,load_checkpoint
from data import CsvMNISTDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Evaluator:
    def __init__(self, config):
        self.val_loader = None
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型、优化器、损失函数
        self.model = LeNet(**config.model.model_args).to(self.device)
        load_checkpoint(self.model,config.train.save_path)
        # 日志 & 其他工具
        self.logger = get_logger()
        self.best_val_acc = 0.0
        self.predictions = []

    def load_data(self):
        # 或者从外部传入 DataLoader，更灵活
        dataset = CsvMNISTDataset(self.config.data.test_csv, has_labels=False)
        self.val_loader = DataLoader(
            dataset,
            batch_size=self.config.eval.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        self.logger.info(f'Dataset length:{len(dataset)}')
        self.logger.info(f"Number of batches: {len(self.val_loader)}")

    @torch.no_grad()
    def evaluate(self):
        self.logger.info("开始进行测试")
        self.load_data()
        self.model.eval()
        predictions = []
        for batch in self.val_loader:
            inputs = batch[0].to(self.device)  # 假设 batch 是 (image,)
            outputs = F.softmax(self.model(inputs),-1)
            preds = outputs.argmax(1)
            predictions.extend(preds.cpu().numpy())

        self.logger.info("测试结束，开始存储结果")
        self.predictions = predictions
        self.save_result()

    def save_result(self):
        df = pd.DataFrame({"ImageId": range(1,len(self.predictions)+1), "Label": self.predictions})
        df.to_csv(self.config.data.root+"/submission.csv", index=False)