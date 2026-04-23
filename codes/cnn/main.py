from utils import load_config,setup_environment
from train import Trainer
from evaluate import Evaluator

def train():
    config = load_config("configs/config.yaml")

    # 可选：设置随机种子、设备等全局配置
    setup_environment(config)

    trainer = Trainer(config)
    trainer.train()


def evaluate():
    config = load_config("configs/config.yaml")

    # 可选：设置随机种子、设备等全局配置
    setup_environment(config)

    evaluator = Evaluator(config)
    evaluator.evaluate()

if __name__ == "__main__":
    # train()
    train()