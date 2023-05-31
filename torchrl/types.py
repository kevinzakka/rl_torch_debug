"""Type definitions."""

from typing import NamedTuple

import torch


class Transition(NamedTuple):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_observation: torch.Tensor
