from typing import Callable, Sequence, Type, TypeAlias

import torch
import torch.nn as nn

BaseCls: TypeAlias = Callable[..., nn.Module]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: Type[nn.Module] = nn.ReLU,
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self._layers = nn.Sequential()
        for i, size in enumerate(hidden_dims):
            self._layers.append(nn.Linear(input_dim, size))
            if dropout_rate > 0:
                self._layers.append(nn.Dropout(dropout_rate))
            if use_layer_norm:
                self._layers.append(nn.LayerNorm(size))
            if i < len(hidden_dims) - 1 or activate_final:
                self._layers.append(activation())
            input_dim = size

        # Weight initialization.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class StateActionValue(nn.Module):
    def __init__(self, base_cls: BaseCls, base_cls_output_dim: int) -> None:
        super().__init__()

        self._base_cls = base_cls()
        self._value = nn.Linear(base_cls_output_dim, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        out = self._base_cls(inputs, *args, **kwargs)
        value = self._value(out)
        return torch.squeeze(value, -1)


class Ensemble(nn.Module):
    def __init__(self, net_cls: BaseCls, num: int = 2) -> None:
        super().__init__()

        self._ensemble = nn.ModuleList()
        for _ in range(num):
            self._ensemble.append(net_cls())

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.stack([net(*args, **kwargs) for net in self._ensemble], dim=0)
