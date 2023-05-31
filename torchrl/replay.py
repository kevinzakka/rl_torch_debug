"""Replay buffer module."""

from typing import Optional
import torch
import numpy as np
import dm_env

from torchrl.specs import EnvironmentSpec
from torchrl import types


class ReplayBuffer:
    """A replay buffer that stores torch tensors on GPU memory."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        spec: EnvironmentSpec,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._capacity: int = capacity
        self._batch_size: int = batch_size
        self._spec = spec
        self._device = device

        self._index: int = 0
        self._size: int = 0
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None

        self._observations = torch.empty(
            (capacity, *spec.observation.shape), dtype=torch.float32, device=device
        )
        self._actions = torch.empty(
            (capacity, *spec.action.shape), dtype=torch.float32, device=device
        )
        self._rewards = torch.empty((capacity), dtype=torch.float32, device=device)
        self._discounts = torch.empty((capacity), dtype=torch.float32, device=device)
        self._next_observations = torch.empty(
            (capacity, *spec.observation.shape), dtype=torch.float32, device=device
        )

    def __len__(self) -> int:
        return self._size

    def is_ready(self) -> bool:
        return self._batch_size <= len(self)

    def insert(self, timestep: dm_env.TimeStep, action: Optional[np.ndarray]) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self._observations[self._index] = torch.from_numpy(
                self._prev.observation  # type: ignore
            ).float()
            self._actions[self._index] = torch.from_numpy(self._action).float()
            self._rewards[self._index] = torch.from_numpy(
                np.asarray(self._latest.reward)
            ).float()
            self._discounts[self._index] = torch.from_numpy(
                np.asarray(self._latest.discount)
            ).float()
            self._next_observations[self._index] = torch.from_numpy(
                self._latest.observation
            ).float()

            self._size = min(self._size + 1, self._capacity)
            self._index = (self._index + 1) % self._capacity

    def sample(self) -> types.Transition:
        indices = torch.randint(
            low=0, high=self._size, size=(self._batch_size,), dtype=torch.long
        )

        return types.Transition(
            observation=self._observations[indices],
            action=self._actions[indices],
            reward=self._rewards[indices],
            discount=self._discounts[indices],
            next_observation=self._next_observations[indices],
        )
