from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InferenceResult:
    image_scores: np.ndarray
    image_targets: np.ndarray
    pixel_scores: np.ndarray
    pixel_targets: np.ndarray
    masks: list[np.ndarray]


def normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    return (scores - lo) / (hi - lo + eps)


def best_overlap_metrics(target: np.ndarray, score: np.ndarray, steps: int = 101) -> dict:
    thresholds = np.linspace(0.0, 1.0, steps)
    best = {"f1": 0.0, "iou": 0.0, "threshold": 0.0}
    target = target.astype(bool)
    score = normalize_scores(score)
    for threshold in thresholds:
        pred = score >= threshold
        tp = np.logical_and(pred, target).sum()
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = tp / max(tp + fp + fn, 1)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "iou": float(iou), "threshold": float(threshold)}
    return best


def image_level_score(prob: torch.Tensor, pool_size: int = 21) -> torch.Tensor:
    pooled = F.avg_pool2d(prob, pool_size, stride=1, padding=pool_size // 2)
    return pooled.flatten(1).max(dim=1).values


class MaskPostProcessor:
    def __init__(self, threshold: float | None = None, blur_size: int = 0):
        self.threshold = threshold
        self.blur_size = blur_size

    def __call__(self, prob: torch.Tensor) -> torch.Tensor:
        mask = prob
        if self.blur_size and self.blur_size > 1:
            mask = F.avg_pool2d(mask, self.blur_size, stride=1, padding=self.blur_size // 2)
        if self.threshold is not None:
            mask = (mask >= self.threshold).float()
        return mask


class LocalizationEvaluator:
    def __init__(self, model, loader, device: torch.device | str = "cuda", postprocess: MaskPostProcessor | None = None):
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.postprocess = postprocess or MaskPostProcessor()

    @torch.no_grad()
    def collect(self) -> InferenceResult:
        self.model.eval()
        image_scores = []
        image_targets = []
        pixel_scores = []
        pixel_targets = []
        masks = []

        for batch in self.loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            target = batch["has_anomaly"]
            logits = self.model(image)
            prob = F.softmax(logits, dim=1)[:, 1:2]
            score = image_level_score(prob)
            pred_mask = self.postprocess(prob)

            image_scores.extend(score.detach().cpu().numpy().tolist())
            image_targets.extend(np.asarray(target).reshape(-1).tolist())
            pixel_scores.append(prob.detach().cpu().numpy().reshape(-1))
            pixel_targets.append(mask.detach().cpu().numpy().reshape(-1))
            masks.extend(pred_mask.detach().cpu().numpy()[:, 0])

        return InferenceResult(
            image_scores=np.asarray(image_scores, dtype=np.float32),
            image_targets=np.asarray(image_targets, dtype=np.float32),
            pixel_scores=np.concatenate(pixel_scores) if pixel_scores else np.asarray([], dtype=np.float32),
            pixel_targets=np.concatenate(pixel_targets) if pixel_targets else np.asarray([], dtype=np.float32),
            masks=masks,
        )

    def evaluate(self) -> dict:
        result = self.collect()
        overlap = best_overlap_metrics(result.pixel_targets, result.pixel_scores)
        return {
            "image_scores": result.image_scores,
            "image_targets": result.image_targets,
            "pixel_scores": result.pixel_scores,
            "pixel_targets": result.pixel_targets,
            "pixel_f1": overlap["f1"],
            "pixel_iou": overlap["iou"],
            "threshold": overlap["threshold"],
        }
