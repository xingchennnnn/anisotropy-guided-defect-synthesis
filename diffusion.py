from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ObjectiveWeights:
    noise: float = 1.0
    mask: float = 1.0
    perceptual: float = 0.1
    color: float = 0.1
    structure: float = 0.2


def extract_at(v: torch.Tensor, index: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = torch.gather(v, dim=0, index=index).to(index.device, dtype=torch.float32)
    return out.reshape([index.shape[0]] + [1] * (len(shape) - 1))


class LinearNoiseSchedule(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 1e-2):
        super().__init__()
        self.timesteps = int(timesteps)
        beta = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("signal", torch.sqrt(alpha_bar))
        self.register_buffer("noise", torch.sqrt(1.0 - alpha_bar))

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0) if eps is None else eps
        xt = extract_at(self.signal, t, x0.shape) * x0 + extract_at(self.noise, t, x0.shape) * eps
        return xt, eps

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (xt - extract_at(self.noise, t, xt.shape) * eps) / extract_at(self.signal, t, xt.shape).clamp_min(1e-8)

    def reverse_step(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        x0 = self.predict_x0(xt, t, eps).clamp(0.0, 1.0)
        prev_t = torch.clamp(t - 1, min=0)
        return extract_at(self.signal, prev_t, xt.shape) * x0 + extract_at(self.noise, prev_t, xt.shape) * eps


class RegionLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target.float())
        numerator = 2.0 * (pred * target).sum(dim=(1, 2, 3))
        denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        dice = 1.0 - (numerator / denominator).mean()
        return 0.5 * bce + 0.5 * dice


class StructureWeightedObjective(nn.Module):
    def __init__(self, weights: ObjectiveWeights | None = None):
        super().__init__()
        self.weights = weights or ObjectiveWeights()
        self.region_loss = RegionLoss()

    def structure_weight(self, structure_map: torch.Tensor | None, mask_pred: torch.Tensor) -> torch.Tensor:
        if structure_map is not None and structure_map.shape[1] >= 5:
            coherence = structure_map[:, 4:5].clamp(0.0, 1.0)
        else:
            coherence = 1.0 - mask_pred.detach()
        return 1.0 + 2.0 * (1.0 - coherence)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        mask_pred: torch.Tensor,
        mask_target: torch.Tensor,
        x0_pred: torch.Tensor,
        x0_target: torch.Tensor,
        structure_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        weight = self.structure_weight(structure_map, mask_pred).repeat(1, eps_pred.shape[1], 1, 1)
        noise = (((eps_pred - eps_target) ** 2) * weight).mean()
        region = self.region_loss(mask_pred, mask_target)
        perceptual_proxy = F.l1_loss(x0_pred, x0_target)
        color_proxy = F.mse_loss(mask_target * x0_pred, mask_target * x0_target)
        structure_proxy = total_variation(mask_pred * self.structure_weight(structure_map, mask_pred))
        loss = (
            self.weights.noise * noise
            + self.weights.mask * region
            + self.weights.perceptual * perceptual_proxy
            + self.weights.color * color_proxy
            + self.weights.structure * structure_proxy
        )
        return loss, {
            "noise": noise.detach(),
            "mask": region.detach(),
            "perceptual": perceptual_proxy.detach(),
            "color": color_proxy.detach(),
            "structure": structure_proxy.detach(),
        }


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


class ConditionalDiffusionCore(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        structure_encoder: nn.Module | None = None,
        schedule: LinearNoiseSchedule | None = None,
        objective: StructureWeightedObjective | None = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.structure_encoder = structure_encoder
        self.schedule = schedule or LinearNoiseSchedule()
        self.objective = objective or StructureWeightedObjective()

    def encode_structure(self, image: torch.Tensor, mask: torch.Tensor, meta: dict | None):
        if meta is not None and "precomputed_direction_map" in meta:
            return meta["precomputed_direction_map"].to(image.device)
        if self.structure_encoder is None:
            return None
        return self.structure_encoder(image, mask, meta or {})

    def training_step(self, image: torch.Tensor, mask: torch.Tensor, meta: dict | None = None, class_labels=None):
        t = self.schedule.sample_t(image.shape[0], image.device)
        xt, eps = self.schedule.add_noise(image, t)
        structure_map = self.encode_structure(image, mask, meta)
        eps_pred, mask_pred = self.denoiser(
            xt,
            t,
            mask,
            anomaly_img=image,
            given_structure_map=structure_map,
            class_labels=class_labels,
        )
        x0_pred = self.schedule.predict_x0(xt, t, eps_pred).clamp(0.0, 1.0)
        return self.objective(eps_pred, eps, mask_pred, mask, x0_pred, image, structure_map)


class TransferSampler:
    def __init__(self, denoiser: nn.Module, schedule: LinearNoiseSchedule, steps: Iterable[int] | None = None):
        self.denoiser = denoiser
        self.schedule = schedule
        self.steps = list(steps) if steps is not None else list(range(schedule.timesteps - 1, -1, -1))

    @torch.no_grad()
    def sample(self, image: torch.Tensor, mask: torch.Tensor, structure_map: torch.Tensor | None = None, class_labels=None) -> torch.Tensor:
        latent = torch.randn_like(image)
        for step in self.steps:
            t = torch.full((image.shape[0],), int(step), device=image.device, dtype=torch.long)
            eps, _ = self.denoiser(
                latent,
                t,
                mask,
                anomaly_img=image,
                given_structure_map=structure_map,
                class_labels=class_labels,
            )
            latent = self.schedule.reverse_step(latent, t, eps)
        return (1.0 - mask) * image + mask * latent.clamp(0.0, 1.0)
