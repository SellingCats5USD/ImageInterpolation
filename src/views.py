from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class View:
    name: str
    forward: TensorTransform
    inverse: TensorTransform


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2,))


def hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-1,))


def rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2, -1))


SUPPORTED_VIEWS: dict[str, View] = {
    "identity": View("identity", identity, identity),
    "vflip": View("vflip", vflip, vflip),
    "hflip": View("hflip", hflip, hflip),
    "rot180": View("rot180", rot180, rot180),
}


def get_view(name: str) -> View:
    if name not in SUPPORTED_VIEWS:
        options = ", ".join(sorted(SUPPORTED_VIEWS.keys()))
        raise ValueError(f"Unknown view '{name}'. Supported views: {options}")
    return SUPPORTED_VIEWS[name]
