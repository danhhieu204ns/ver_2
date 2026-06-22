#!/usr/bin/env python3
"""Shared, bbox-safe training augmentation for detection experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class DetectionAugmentation:
    """Photometric augmentation plus horizontal flip; evaluation uses zeros."""

    hflip_prob: float = 0.0
    brightness: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hflip_prob <= 1.0:
            raise ValueError("hflip_prob must be in [0, 1]")
        if not 0.0 <= self.brightness <= 1.0:
            raise ValueError("brightness must be in [0, 1]")
        if not 0.0 <= self.saturation <= 1.0:
            raise ValueError("saturation must be in [0, 1]")
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5]")

    @property
    def enabled(self) -> bool:
        return any((self.hflip_prob, self.brightness, self.saturation, self.hue))


def _symmetric_factor(magnitude: float) -> float:
    return random.uniform(max(0.0, 1.0 - magnitude), 1.0 + magnitude)


def augment_pil_detection(
    image: Image.Image,
    boxes_xyxy: list[tuple[float, float, float, float]],
    config: DetectionAugmentation,
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """Apply the canonical policy and return transformed XYXY boxes."""

    if not config.enabled:
        return image, boxes_xyxy

    operations: list[tuple[str, float]] = []
    if config.brightness > 0:
        operations.append(("brightness", _symmetric_factor(config.brightness)))
    if config.saturation > 0:
        operations.append(("saturation", _symmetric_factor(config.saturation)))
    if config.hue > 0:
        operations.append(("hue", random.uniform(-config.hue, config.hue)))
    random.shuffle(operations)

    for name, factor in operations:
        if name == "brightness":
            image = F.adjust_brightness(image, factor)
        elif name == "saturation":
            image = F.adjust_saturation(image, factor)
        else:
            image = F.adjust_hue(image, factor)

    if config.hflip_prob > 0 and random.random() < config.hflip_prob:
        width = image.width
        image = F.hflip(image)
        boxes_xyxy = [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes_xyxy]

    return image, boxes_xyxy
