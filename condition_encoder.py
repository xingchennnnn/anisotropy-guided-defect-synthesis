from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from aniso_fields import build_aniso_structure_map


@dataclass
class StructureEncoderConfig:
    input_channels: int = 5
    embed_channels: int = 256
    hidden_ratio: float = 0.5
    patch_size: int = 16
    normalize_input: bool = True
    mask_external_map: bool = True


class StructureMapAdapter(nn.Module):
    def __init__(self, target_channels: int = 5):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, structure_map: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if structure_map.dim() == 3:
            structure_map = structure_map.unsqueeze(0)
        if structure_map.shape[1] > self.target_channels:
            structure_map = structure_map[:, : self.target_channels]
        if structure_map.shape[1] < self.target_channels:
            pad = self.target_channels - structure_map.shape[1]
            structure_map = F.pad(structure_map, (0, 0, 0, 0, 0, pad))
        if mask is not None and mask.shape[-2:] != structure_map.shape[-2:]:
            mask = F.interpolate(mask.float(), size=structure_map.shape[-2:], mode="nearest")
        return structure_map if mask is None else structure_map * mask


class StructureMapNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None or mask.sum() <= 0:
            scale = x.flatten(2).std(dim=-1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(self.eps)
            return x / scale
        valid = mask > 0.5
        out = x.clone()
        for b in range(x.shape[0]):
            region = valid[b].expand_as(x[b])
            if region.any():
                values = x[b][region]
                out[b] = (x[b] - values.mean()) / values.std().clamp_min(self.eps)
        return out * mask


class AnisotropicStructureEncoder(nn.Module):
    def __init__(self, embed_channels: int = 256, patch_size: int = 16, config: StructureEncoderConfig | None = None):
        super().__init__()
        self.config = config or StructureEncoderConfig(embed_channels=embed_channels, patch_size=patch_size)
        hidden = max(8, int(self.config.embed_channels * self.config.hidden_ratio))
        self.adapter = StructureMapAdapter(self.config.input_channels)
        self.normalizer = StructureMapNorm()
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.input_channels, hidden, 3, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden, self.config.embed_channels, 3, padding=1),
            nn.GroupNorm(min(16, self.config.embed_channels), self.config.embed_channels),
            nn.SiLU(),
            nn.Conv2d(self.config.embed_channels, self.config.embed_channels, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(self.config.input_channels, self.config.embed_channels, 1),
            nn.Sigmoid(),
        )

    def make_structure_map(
        self,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
        aniso_config: dict | None = None,
        given_structure_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if given_structure_map is None:
            structure_map = build_aniso_structure_map(anomaly_img, mask, aniso_config)
        else:
            use_mask = mask if self.config.mask_external_map else None
            structure_map = self.adapter(given_structure_map, use_mask)
        if self.config.normalize_input:
            structure_map = self.normalizer(structure_map, mask)
        return structure_map

    def forward(
        self,
        anomaly_img: torch.Tensor,
        mask: torch.Tensor,
        aniso_config: dict | None = None,
        given_structure_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        structure_map = self.make_structure_map(anomaly_img, mask, aniso_config, given_structure_map)
        feature = self.body(self.stem(structure_map))
        return feature * (1.0 + self.gate(structure_map))
