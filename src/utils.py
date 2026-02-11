from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def make_generator(seed: int, device: str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def save_pil(image: Image.Image, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_horizontal_grid(images: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(img.size for img in images))
    canvas = Image.new("RGB", (int(np.sum(widths)), int(np.max(heights))))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas
