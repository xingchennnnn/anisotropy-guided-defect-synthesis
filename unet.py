from __future__ import annotations

import math
from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from condition_encoder import AnisotropicStructureEncoder
from cross_attention import AnisotropicCrossAttentionBlock, LightweightAnisotropicCrossAttention


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def norm_layer(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PositionalMaskEmbedding(nn.Module):
    def __init__(self, embed_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, embed_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embed_channels, embed_channels, 3, padding=1),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(mask)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask_emb: torch.Tensor | None = None, structure_emb: torch.Tensor | None = None):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask_emb: torch.Tensor | None = None, structure_emb: torch.Tensor | None = None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, mask_emb, structure_emb)
            else:
                x = layer(x)
        return x


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, dropout: float, mask_channels: int, structure_channels: int):
        super().__init__()
        self.norm1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.mask_proj = nn.Conv2d(mask_channels, out_channels, 1)
        self.structure_proj = nn.Sequential(nn.Conv2d(structure_channels, out_channels * 2, 1), nn.SiLU())
        self.norm2 = norm_layer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor, mask_emb: torch.Tensor | None = None, structure_emb: torch.Tensor | None = None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(emb).unsqueeze(-1).unsqueeze(-1)
        if mask_emb is not None:
            mask_emb = F.interpolate(mask_emb, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + self.mask_proj(mask_emb)
        if structure_emb is not None:
            structure_emb = F.interpolate(structure_emb, size=h.shape[-2:], mode="bilinear", align_corners=False)
            gate, shift = torch.chunk(self.structure_proj(structure_emb), 2, dim=1)
            h = (1.0 + torch.tanh(gate)) * h + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(self.head_dim)
        h = torch.einsum("bhnm,bhdm->bhdn", attn.softmax(dim=-1), v).reshape(batch, channels, height, width)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8, 16, 32),
        dropout: float = 0.0,
        channel_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4),
        conv_resample: bool = True,
        num_heads: int = 4,
        num_classes: int | None = None,
        direction_cross_attention: dict | None = None,
        image_size: int = 256,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_embed = nn.Embedding(num_classes, time_dim) if num_classes is not None else None
        self.mask_embed = PositionalMaskEmbedding(model_channels)
        self.structure_encoder = AnisotropicStructureEncoder(model_channels, patch_size=16)

        cross_cfg = direction_cross_attention or {}
        self.use_cross_attention = cross_cfg.get("enable", True) and "bottleneck" in cross_cfg.get("positions", ["bottleneck"])
        self.cross_attention_type = cross_cfg.get("attention_type", "full")

        self.down_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))])
        skip_channels = [model_channels]
        ch = model_channels
        current_res = image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, model_channels * mult, time_dim, dropout, model_channels, model_channels)]
                ch = model_channels * mult
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                skip_channels.append(ch)
                current_res //= 2

        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_dim, dropout, model_channels, model_channels),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_dim, dropout, model_channels, model_channels),
        )
        if self.use_cross_attention:
            if self.cross_attention_type == "lightweight":
                self.structure_cross_attn = LightweightAnisotropicCrossAttention(
                    ch,
                    model_channels,
                    cross_cfg.get("num_heads", num_heads),
                    cross_cfg.get("dropout", dropout),
                    cross_cfg.get("lightweight_config", {}).get("downsample_ratio", 2),
                )
            else:
                self.structure_cross_attn = AnisotropicCrossAttentionBlock(
                    ch,
                    model_channels,
                    cross_cfg.get("num_heads", num_heads),
                    cross_cfg.get("dropout", dropout),
                    cross_cfg.get("ffn_expansion", 4),
                )
        else:
            self.structure_cross_attn = None

        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for block_id in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch + skip_channels.pop(), model_channels * mult, time_dim, dropout, model_channels, model_channels)]
                ch = model_channels * mult
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level and block_id == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    current_res *= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(norm_layer(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))
        self.mask_head = nn.Sequential(
            nn.Conv2d(ch, max(ch // 2, 1), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(ch // 2, 1), 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor | None,
        anomaly_img: torch.Tensor | None = None,
        aniso_config: dict | None = None,
        given_structure_map: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.class_embed is not None:
            if class_labels is None:
                raise ValueError("class_labels are required when num_classes is set")
            emb = emb + self.class_embed(class_labels)

        mask_emb = self.mask_embed(mask)
        structure_emb = None
        if anomaly_img is not None and mask.sum() > 0:
            structure_emb = self.structure_encoder(anomaly_img, mask, aniso_config, given_structure_map)

        hs = []
        h = x
        for block in self.down_blocks:
            h = block(h, emb, mask_emb, structure_emb)
            hs.append(h)
        h = self.middle_block(h, emb, mask_emb, structure_emb)
        if self.structure_cross_attn is not None and structure_emb is not None:
            h = self.structure_cross_attn(h, structure_emb)
        for block in self.up_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb, mask_emb, structure_emb)
        return self.out(h), self.mask_head(h)
