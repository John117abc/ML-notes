import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    多头注意力实现
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 1. 维度校验
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 计算每个头内部特征向量的维度

        # 2. 创建 Query, Key, Value 的线性投影层
        # 将输入维度 d_in 映射到输出维度 d_out (所有头的总维度)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 3. 输出投影层 (Optional Projection)
        # 将多头拼接后的结果映射回 d_out 维度
        self.out_proj = nn.Linear(d_out, d_out)

        # 4. 随机失活 (Dropout)
        self.dropout = nn.Dropout(dropout)

        # 5. 注册因果掩码 (Causal Mask)
        # 这是一个上三角矩阵，用于在自注意力中屏蔽未来信息
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 输入 x 的形状是 (batch_size, num_tokens, d_in)。
        # 经过线性层后，形状变为 (batch_size, num_tokens, d_out)。
        # 注意：这里并没有立即拆分成多个头，而是先合并成一个大的向量。
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 这是多头注意力最关键的一步，目的是将“批量数据”、“token序列”、“多头”和“头内维度”分开
        # 重塑：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：将 heads 维度移动到第二个位置，方便后续矩阵运算
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算点积: (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        # 结果形状: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 获取掩码布尔值，截取到当前实际token数量
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 将未来位置（上三角部分）填充为负无穷，softmax 后概率为 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放并 Softmax
        # keys.shape[-1] 即 head_dim
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        # 结果形状: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 拼接：将多个头的结果拼接到最后一个维度
        # (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 可选的线性投影层
        context_vec = self.out_proj(context_vec)
        return context_vec