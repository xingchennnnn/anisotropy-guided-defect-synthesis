from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


DEFAULT_BASE_CHANNELS = 40


@dataclass
class CheckpointConfig:
    in_channels: int = 3
    out_channels: int = 2
    base_channels: int = DEFAULT_BASE_CHANNELS


def conv_block(in_channels: int, out_channels: int, use_norm: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        c = base_channels
        self.block1 = conv_block(in_channels, c)
        self.block2 = conv_block(c, c * 2)
        self.block3 = conv_block(c * 2, c * 4)
        self.block4 = conv_block(c * 4, c * 8)
        self.block5 = conv_block(c * 8, c * 8)
        self.block6 = conv_block(c * 8, c * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b1 = self.block1(x)
        b2 = self.block2(self.pool(b1))
        b3 = self.block3(self.pool(b2))
        b4 = self.block4(self.pool(b3))
        b5 = self.block5(self.pool(b4))
        b6 = self.block6(self.pool(b5))
        return b1, b2, b3, b4, b5, b6


class Decoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c = base_channels
        self.up5 = up_block(c * 8, c * 8)
        self.dec5 = conv_block(c * 16, c * 8)
        self.up4 = up_block(c * 8, c * 4)
        self.dec4 = conv_block(c * 12, c * 4)
        self.up3 = up_block(c * 4, c * 2)
        self.dec3 = conv_block(c * 6, c * 2)
        self.up2 = up_block(c * 2, c)
        self.dec2 = conv_block(c * 3, c)
        self.up1 = up_block(c, c)
        self.dec1 = conv_block(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, 3, padding=1)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        b1, b2, b3, b4, b5, b6 = features
        x = self.dec5(torch.cat([self.up5(b6), b5], dim=1))
        x = self.dec4(torch.cat([self.up4(x), b4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), b3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), b2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), b1], dim=1))
        return self.head(x)


class SegmentationUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        return_features: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels, out_channels)
        self.return_features = return_features

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        logits = self.decoder(features)
        if self.return_features:
            return logits, features
        return logits


class ReconstructionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 96):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = ReconstructionDecoder(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x)[-1])


class ReconstructionDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            up_block(c * 8, c * 8),
            conv_block(c * 8, c * 8),
            up_block(c * 8, c * 4),
            conv_block(c * 4, c * 4),
            up_block(c * 4, c * 2),
            conv_block(c * 2, c * 2),
            up_block(c * 2, c),
            conv_block(c, c),
            up_block(c, c),
            nn.Conv2d(c, out_channels, 3, padding=1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.net(bottleneck)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def infer_base_channels(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("encoder.block1.0.weight", "encoder_segment.block1.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError("base channel key not found")


def build_checkpoint_payload(model: nn.Module, base_channels: int) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": CheckpointConfig(base_channels=int(base_channels)).__dict__,
    }


def unpack_checkpoint(checkpoint) -> tuple[dict, CheckpointConfig]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        raw_config = checkpoint.get("model_config", {})
        config = CheckpointConfig(
            in_channels=int(raw_config.get("in_channels", 3)),
            out_channels=int(raw_config.get("out_channels", 2)),
            base_channels=int(raw_config.get("base_channels", infer_base_channels(state))),
        )
        return state, config
    config = CheckpointConfig(base_channels=infer_base_channels(checkpoint))
    return checkpoint, config


def build_model_from_state(checkpoint) -> SegmentationUNet:
    state, config = unpack_checkpoint(checkpoint)
    model = SegmentationUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
    )
    model.load_state_dict(state, strict=False)
    return model
