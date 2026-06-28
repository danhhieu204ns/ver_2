#!/usr/bin/env python3
"""Shared bbox-safe augmentation for object detection."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageFilter
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class DetectionAugmentation:

    hflip_prob: float = 0.5
    vflip_prob: float = 0.0

    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05

    gamma: float = 0.2
    sharpness: float = 0.3

    blur_prob: float = 0.15
    grayscale_prob: float = 0.05
    jpeg_prob: float = 0.15

    def __post_init__(self):

        assert 0 <= self.hflip_prob <= 1
        assert 0 <= self.vflip_prob <= 1

        assert 0 <= self.brightness <= 1
        assert 0 <= self.contrast <= 1
        assert 0 <= self.saturation <= 1
        assert 0 <= self.hue <= 0.5

        assert 0 <= self.gamma <= 1
        assert 0 <= self.sharpness <= 1

        assert 0 <= self.blur_prob <= 1
        assert 0 <= self.grayscale_prob <= 1
        assert 0 <= self.jpeg_prob <= 1

    @property
    def enabled(self):

        return any(
            (
                self.hflip_prob,
                self.vflip_prob,
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue,
                self.gamma,
                self.sharpness,
                self.blur_prob,
                self.grayscale_prob,
                self.jpeg_prob,
            )
        )


def _factor(mag: float):

    return random.uniform(max(0.0, 1 - mag), 1 + mag)


def _jpeg_compress(image: Image.Image):

    quality = random.randint(45, 90)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    return Image.open(buf).convert("RGB")


def augment_pil_detection(
    image: Image.Image,
    boxes_xyxy: list[tuple[float, float, float, float]],
    config: DetectionAugmentation,
):

    if not config.enabled:
        return image, boxes_xyxy

    ops = []

    if config.brightness > 0:
        ops.append(("brightness", _factor(config.brightness)))

    if config.contrast > 0:
        ops.append(("contrast", _factor(config.contrast)))

    if config.saturation > 0:
        ops.append(("saturation", _factor(config.saturation)))

    if config.hue > 0:
        ops.append(("hue", random.uniform(-config.hue, config.hue)))

    if config.gamma > 0:
        ops.append(("gamma", _factor(config.gamma)))

    if config.sharpness > 0:
        ops.append(("sharpness", _factor(config.sharpness)))

    random.shuffle(ops)

    for name, factor in ops:

        if name == "brightness":
            image = F.adjust_brightness(image, factor)

        elif name == "contrast":
            image = F.adjust_contrast(image, factor)

        elif name == "saturation":
            image = F.adjust_saturation(image, factor)

        elif name == "hue":
            image = F.adjust_hue(image, factor)

        elif name == "gamma":
            image = F.adjust_gamma(image, factor)

        elif name == "sharpness":
            image = F.adjust_sharpness(image, factor)

    if random.random() < config.blur_prob:
        radius = random.uniform(0.3, 1.2)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    if random.random() < config.grayscale_prob:
        image = F.rgb_to_grayscale(image, num_output_channels=3)

    if random.random() < config.jpeg_prob:
        image = _jpeg_compress(image)

    if config.hflip_prob > 0 and random.random() < config.hflip_prob:

        w = image.width

        image = F.hflip(image)

        boxes_xyxy = [
            (
                w - x2,
                y1,
                w - x1,
                y2,
            )
            for x1, y1, x2, y2 in boxes_xyxy
        ]

    if config.vflip_prob > 0 and random.random() < config.vflip_prob:

        h = image.height

        image = F.vflip(image)

        boxes_xyxy = [
            (
                x1,
                h - y2,
                x2,
                h - y1,
            )
            for x1, y1, x2, y2 in boxes_xyxy
        ]

    return image, boxes_xyxy