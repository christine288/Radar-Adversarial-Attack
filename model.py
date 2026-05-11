"""模型定义：基于 TransformerEncoder 的雷达航迹时序分类器。"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码（Transformer 开源实现常用写法）。"""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:, :t, :]


class RadarTrackTransformer(nn.Module):
    """
    时序 Transformer 分类器（适配雷达航迹）。

    输入: (B, T, C)，其中 T=轨迹点数，C=通道数（默认 12）。
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_classes: int = 6,
        dropout: float = 0.2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        seq_len, feat_dim = input_size
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.input_proj = nn.Linear(self.feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max(self.seq_len + 8, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        # 使用时序平均池化提升变长轨迹鲁棒性
        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        return logits


# 兼容旧代码导入名
RadarTrackCNN = RadarTrackTransformer

