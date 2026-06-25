from __future__ import annotations

from typing import List, Union

import torch
from torch import nn


def build_nn(structure: List[int]) -> nn.Module:
    layers = [nn.Linear(structure[0], structure[1])]
    for i in range(1, len(structure) - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(structure[i], structure[i + 1]))
    return nn.Sequential(*layers)


def bound_loss(v: torch.Tensor, lb: Union[float, torch.Tensor], ub: Union[float, torch.Tensor]) -> torch.Tensor:
    relu = nn.ReLU()
    low_loss = relu(lb - v)
    high_loss = relu(v - ub)
    return torch.mean(low_loss + high_loss)
