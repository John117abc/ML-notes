# utils/checkpoint.py
import torch
import os


def save_checkpoint(model, filepath, optimizer=None, epoch=None, extra_info=None):
    """
    保存模型检查点。

    Args:
        model: 要保存的模型（nn.Module）
        filepath: 保存路径，如 "checkpoints/best.pth"
        optimizer: 可选，优化器状态（用于恢复训练）
        epoch: 当前 epoch
        extra_info: 其他你想保存的信息（如 best_acc）
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, filepath)


def load_checkpoint(model, filepath, optimizer=None, device=None):
    """
    加载模型检查点。

    Args:
        model: 要加载权重的模型（nn.Module）
        filepath: 检查点文件路径
        optimizer: 可选，用于恢复优化器状态（评估时通常不需要）
        device: 指定加载到哪个设备（如 'cpu' 或 'cuda'）

    Returns:
        checkpoint: 完整的 checkpoint 字典（包含 epoch、extra_info 等）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # 自动选择设备（如果未指定）
    map_location = device if device is not None else lambda storage, loc: storage
    checkpoint = torch.load(filepath, map_location=map_location)

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 如果提供了 optimizer 且 checkpoint 中有优化器状态，则加载
    if optimizer and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint