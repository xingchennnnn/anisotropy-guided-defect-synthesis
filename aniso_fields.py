from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    distance_transform_edt = None
    _HAS_SCIPY = False


@dataclass
class PatchAnisoMeta:
    y: int
    x: int
    theta: float
    sigma_parallel: float
    sigma_perpendicular: float
    coherence: float


def gaussian_blur_separable(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    channels = img.shape[1]
    radius = int(max(1, round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel = kernel / (kernel.sum() + 1e-8)
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    img = F.conv2d(img, kx, padding=(0, radius), groups=channels)
    img = F.conv2d(img, ky, padding=(radius, 0), groups=channels)
    return img


def sobel_xy(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = img.shape[1]
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=torch.float32,
        device=img.device,
    ) / 8.0
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    return F.conv2d(img, kx, padding=1, groups=channels), F.conv2d(img, ky, padding=1, groups=channels)


def distance_field(mask: torch.Tensor) -> torch.Tensor:
    mask_bool = mask > 0.5
    batch, _, height, width = mask_bool.shape
    if _HAS_SCIPY:
        maps = []
        for b in range(batch):
            arr = mask_bool[b, 0].detach().cpu().numpy().astype(bool)
            maps.append(torch.from_numpy(distance_transform_edt(arr))[None, None])
        return torch.cat(maps, dim=0).to(device=mask.device, dtype=torch.float32)

    inf = 1e6
    dist = torch.where(mask_bool, torch.full_like(mask, inf), torch.zeros_like(mask))
    for _ in range(height + width):
        next_dist = torch.minimum(dist, -F.max_pool2d(-dist, 3, stride=1, padding=1) + 1.0)
        if torch.allclose(next_dist, dist):
            break
        dist = next_dist
    return dist * mask_bool.float()


def structure_tensor(img: torch.Tensor, mask: torch.Tensor, sigma_g: float, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    if img.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
        gray = (img * weights).sum(dim=1, keepdim=True)
    else:
        gray = img

    gray = gaussian_blur_separable(gray * mask, sigma_g)
    gx, gy = sobel_xy(gray)
    j11 = gaussian_blur_separable(gx * gx, rho)
    j22 = gaussian_blur_separable(gy * gy, rho)
    j12 = gaussian_blur_separable(gx * gy, rho)

    theta = 0.5 * torch.atan2(2.0 * j12, j11 - j22 + 1e-12)
    trace = (j11 + j22).clamp_min(1e-12)
    root = torch.sqrt((j11 - j22) ** 2 + 4.0 * j12 ** 2 + 1e-12)
    lambda_1 = 0.5 * (trace + root)
    lambda_2 = 0.5 * (trace - root)
    coherence = ((lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-12)).clamp(0.0, 1.0)
    return theta[:, 0], coherence[:, 0]


def compute_patch_meta_from_anomaly(
    anomaly_img: torch.Tensor,
    mask: torch.Tensor,
    patch: int = 16,
    sigma_g: float = 1.0,
    rho: float = 1.5,
    k_par: float = 2.5,
    k_perp: float = 0.8,
    beta: float = 1.5,
    min_mask_ratio: float = 0.10,
    coherence_gate: float = 0.10,
) -> List[List[PatchAnisoMeta]]:
    batch, _, height, width = anomaly_img.shape
    theta_map, coherence_map = structure_tensor(anomaly_img, mask, sigma_g, rho)
    dist = distance_field(mask)[:, 0]
    all_meta: List[List[PatchAnisoMeta]] = []

    for b in range(batch):
        sample_meta: List[PatchAnisoMeta] = []
        valid = mask[b, 0] > 0.5
        for y in range(0, height, patch):
            for x in range(0, width, patch):
                ys = slice(y, min(y + patch, height))
                xs = slice(x, min(x + patch, width))
                block_mask = valid[ys, xs]
                if block_mask.float().mean() < min_mask_ratio:
                    continue

                theta_values = theta_map[b, ys, xs][block_mask]
                coherence_values = coherence_map[b, ys, xs][block_mask]
                dist_values = dist[b, ys, xs][block_mask]

                cos_mean = torch.cos(2.0 * theta_values).mean()
                sin_mean = torch.sin(2.0 * theta_values).mean()
                theta = float((0.5 * torch.atan2(sin_mean, cos_mean)).item())
                coherence = float(torch.median(coherence_values).item())
                width_est = float(torch.median(dist_values).item())

                sigma_parallel = max(1.0, k_par * width_est)
                if coherence < coherence_gate:
                    sigma_perpendicular = sigma_parallel
                else:
                    sigma_perpendicular = max(1.0, k_perp * (width_est / (1.0 + beta * coherence)))

                sample_meta.append(PatchAnisoMeta(y, x, theta, sigma_parallel, sigma_perpendicular, coherence))
        all_meta.append(sample_meta)
    return all_meta


def build_aniso_structure_map(
    anomaly_img: torch.Tensor,
    mask: torch.Tensor,
    aniso_config: dict | None = None,
) -> torch.Tensor:
    config = aniso_config or {}
    patch = int(config.get("patch", 16))
    batch, _, height, width = anomaly_img.shape
    maps = torch.zeros(batch, 5, height, width, device=anomaly_img.device, dtype=torch.float32)
    meta = compute_patch_meta_from_anomaly(anomaly_img, mask, **config)

    for b, sample_meta in enumerate(meta):
        for item in sample_meta:
            ys = slice(item.y, min(item.y + patch, height))
            xs = slice(item.x, min(item.x + patch, width))
            maps[b, 0, ys, xs] = math.cos(2.0 * item.theta)
            maps[b, 1, ys, xs] = math.sin(2.0 * item.theta)
            maps[b, 2, ys, xs] = math.log(max(item.sigma_parallel, 1e-6))
            maps[b, 3, ys, xs] = math.log(max(item.sigma_perpendicular, 1e-6))
            maps[b, 4, ys, xs] = item.coherence
    return maps * mask


def transfer_structure_to_mask(
    template_map: torch.Tensor,
    new_mask: torch.Tensor,
    k_par: float = 2.5,
    k_perp: float = 0.8,
    beta: float = 1.5,
    coherence_gate: float = 0.01,
) -> torch.Tensor:
    batch, _, height, width = template_map.shape
    valid = (template_map[:, 0:1].abs() + template_map[:, 1:2].abs()) > 0.01
    if valid.sum() > 0:
        cos_mean = template_map[:, 0:1][valid].mean()
        sin_mean = template_map[:, 1:2][valid].mean()
        coherence = template_map[:, 4:5][valid].mean().clamp(0.0, 1.0)
    else:
        cos_mean = torch.tensor(1.0, device=template_map.device)
        sin_mean = torch.tensor(0.0, device=template_map.device)
        coherence = torch.tensor(0.5, device=template_map.device)

    dist = distance_field(new_mask)
    sigma_parallel = torch.clamp(k_par * dist, min=1.0)
    if float(coherence.item()) < coherence_gate:
        sigma_perpendicular = sigma_parallel
    else:
        sigma_perpendicular = torch.clamp(k_perp * dist / (1.0 + beta * coherence), min=1.0)

    out = torch.zeros(batch, 5, height, width, device=template_map.device)
    out[:, 0:1] = cos_mean
    out[:, 1:2] = sin_mean
    out[:, 2:3] = torch.log(sigma_parallel.clamp_min(1e-6))
    out[:, 3:4] = torch.log(sigma_perpendicular.clamp_min(1e-6))
    out[:, 4:5] = coherence
    return out * new_mask
