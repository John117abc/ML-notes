def MPLtrain(dataloader, model, loss_fn, optimizer,device):
    """
    :param dataloader: 提供批量数据（如 DataLoader 实例）
    :param model: 要训练的神经网络；
    :param loss_fn: 损失函数
    :param optimizer: 优化器。
    :return:
    """
    size = len(dataloader.dataset)      # 获取整个训练集样本总数（用于打印进度）。
    model.train()       #  将模型切换到 训练模式（training mode）。 影响某些层的行为，例如： Dropout：训练时启用，推理时关闭； BatchNorm：训练时用 batch 统计量，推理时用全局统计量。 如果不调用 .train()，模型可能表现异常（尤其用了这些层时）。
    for batch, (X, y) in enumerate(dataloader):   #  每次返回一个 batch 的 (X, y)，X：输入图像。y：标签
        X, y = X.to(device), y.to(device)  # 移到设备
        pred = model(X)     # 调用模型的 forward() 方法，得到 logits（shape [64, 10]），向前传播
        loss = loss_fn(pred, y)     # 计算当前 batch 的平均损失（标量）。

        # 反向传播更新参数
        loss.backward()     # 自动计算损失对所有可学习参数的梯度（通过反向传播算法）。 梯度会累加到每个参数的 .grad 属性中。
        optimizer.step()        # 根据当前梯度（.grad）和优化器规则（如 SGD: w = w - lr * grad）更新参数。
        optimizer.zero_grad()       #  清空梯度缓冲区（将所有 .grad 设为 None 或 0）。 如果不清零，梯度会累积！ 下一次 backward() 会把新梯度加到旧梯度上，导致错误更新。
        if batch % 100 == 0:        # 打印训练进度
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")