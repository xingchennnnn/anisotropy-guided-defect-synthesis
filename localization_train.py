from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_score: float = float("-inf")
    history: list[dict] = field(default_factory=list)


class FocalSegmentationLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, prob: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        target = target_mask.long().squeeze(1).clamp(0, 1)
        pt = prob.gather(1, target[:, None]).clamp(self.eps, 1.0 - self.eps)
        loss = -((1.0 - pt) ** self.gamma) * torch.log(pt)
        if self.alpha is not None:
            weights = torch.where(target[:, None] > 0, self.alpha, 1.0 - self.alpha)
            loss = loss * weights
        return loss.mean()


class DiceTerm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prob: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        pred = prob[:, 1:2]
        target = target_mask.float()
        numerator = 2.0 * (pred * target).sum(dim=(1, 2, 3))
        denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        return 1.0 - (numerator / denominator).mean()


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 0.5, smooth_weight: float = 0.05):
        super().__init__()
        self.focal = FocalSegmentationLoss()
        self.dice = DiceTerm()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth_weight = smooth_weight

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        prob = F.softmax(logits, dim=1)
        focal = self.focal(prob, target_mask)
        dice = self.dice(prob, target_mask)
        smooth = total_variation(prob[:, 1:2])
        loss = self.focal_weight * focal + self.dice_weight * dice + self.smooth_weight * smooth
        return loss, {"focal": focal.detach(), "dice": dice.detach(), "smooth": smooth.detach()}


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dx + dy


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def default_score(metrics: dict) -> float:
    fields = ["image_auc", "image_ap", "pixel_auc", "pixel_ap", "pixel_f1"]
    values = [float(metrics[k]) for k in fields if k in metrics]
    return sum(values) / max(len(values), 1)


class LocalizationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cuda",
        score_fn: Callable[[dict], float] = default_score,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or CompositeSegmentationLoss()
        self.device = torch.device(device)
        self.score_fn = score_fn
        self.state = TrainState()

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        totals = {}
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, self.device)
            image = batch["image"]
            mask = batch["mask"]
            logits = self.model(image)
            loss, parts = self.criterion(logits, mask)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            count += 1
            self.state.step += 1

        if self.scheduler is not None:
            self.scheduler.step()
        return {key: value / max(count, 1) for key, value in totals.items()}

    def fit(self, train_loader, evaluator=None, epochs: int = 1) -> TrainState:
        for _ in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            record = {"epoch": self.state.epoch, **train_metrics}
            if evaluator is not None:
                eval_metrics = evaluator.evaluate()
                score = self.score_fn(eval_metrics)
                record.update(eval_metrics)
                record["score"] = score
                if score > self.state.best_score:
                    self.state.best_score = score
                    record["selected"] = True
            self.state.history.append(record)
            self.state.epoch += 1
        return self.state
