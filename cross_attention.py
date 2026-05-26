from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class AnisotropicCrossAttention(nn.Module):
    def __init__(self, feature_channels: int, aniso_channels: int | None = None, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        aniso_channels = aniso_channels or feature_channels
        if feature_channels % num_heads != 0:
            raise ValueError("feature_channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.norm_feature = _group_norm(feature_channels)
        self.norm_aniso = _group_norm(aniso_channels)
        self.query_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        self.key_proj = nn.Conv2d(aniso_channels, feature_channels, 1)
        self.value_proj = nn.Conv2d(aniso_channels, feature_channels, 1)
        self.out_proj = nn.Conv2d(feature_channels, feature_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature: torch.Tensor, aniso_emb: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        feature_norm = self.norm_feature(feature)
        aniso_norm = self.norm_aniso(aniso_emb)
        if aniso_norm.shape[-2:] != feature_norm.shape[-2:]:
            aniso_norm = F.interpolate(aniso_norm, size=feature_norm.shape[-2:], mode="bilinear", align_corners=False)

        q = self.query_proj(feature_norm).reshape(batch, self.num_heads, self.head_dim, height * width)
        k = self.key_proj(aniso_norm).reshape(batch, self.num_heads, self.head_dim, height * width)
        v = self.value_proj(aniso_norm).reshape(batch, self.num_heads, self.head_dim, height * width)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(batch, channels, height, width)
        return feature + self.out_proj(out)


class AnisotropicCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        aniso_channels: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.cross_attn = AnisotropicCrossAttention(feature_channels, aniso_channels, num_heads, dropout)
        hidden_dim = feature_channels * ffn_expansion
        self.ffn = nn.Sequential(
            _group_norm(feature_channels),
            nn.Conv2d(feature_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, feature_channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, feature: torch.Tensor, aniso_emb: torch.Tensor) -> torch.Tensor:
        feature = self.cross_attn(feature, aniso_emb)
        return feature + self.ffn(feature)


class LightweightAnisotropicCrossAttention(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        aniso_channels: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        downsample_ratio: int = 2,
    ):
        super().__init__()
        self.downsample = nn.AvgPool2d(downsample_ratio) if downsample_ratio > 1 else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=downsample_ratio, mode="bilinear", align_corners=False) if downsample_ratio > 1 else nn.Identity()
        self.attn = AnisotropicCrossAttention(feature_channels, aniso_channels, num_heads, dropout)

    def forward(self, feature: torch.Tensor, aniso_emb: torch.Tensor) -> torch.Tensor:
        down_feature = self.downsample(feature)
        down_aniso = self.downsample(aniso_emb)
        out = self.attn(down_feature, down_aniso)
        return feature + self.upsample(out - down_feature)
