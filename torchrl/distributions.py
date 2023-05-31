from typing import TypeAlias, Callable
import torch
from torch import nn
from torch import distributions as dists
import functools

BaseCls: TypeAlias = Callable[..., nn.Module]
TanhTransform = functools.partial(dists.TanhTransform, cache_size=1)


class TanhDiagNormal(dists.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__(dists.Normal(loc, scale), TanhTransform())

    def mode(self) -> torch.Tensor:
        mode = self.base_dist.mode
        for tr in self.transforms:
            mode = tr(mode)
        return mode


class TanhNormal(nn.Module):
    def __init__(
        self,
        base_cls: BaseCls,
        base_cls_output_dim: int,
        action_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        squash_tanh: bool = False,
    ) -> None:
        super().__init__()

        self._base_cls = base_cls()
        self._mean = nn.Linear(base_cls_output_dim, action_dim)
        self._log_std = nn.Linear(base_cls_output_dim, action_dim)

        nn.init.xavier_uniform_(self._mean.weight)
        nn.init.zeros_(self._mean.bias)
        nn.init.xavier_uniform_(self._log_std.weight)
        nn.init.zeros_(self._log_std.bias)

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._squash_tanh = squash_tanh

    def forward(self, inputs, *args, **kwargs) -> dists.Distribution:
        x = self._base_cls(inputs, *args, **kwargs)

        means = self._mean(x)
        log_stds = self._log_std(x)

        log_stds = torch.clamp(log_stds, self._log_std_min, self._log_std_max)
        stds = torch.exp(log_stds)

        if self._squash_tanh:
            return TanhDiagNormal(means, stds)
        # TODO(kevin): This might be wrong.
        return dists.Normal(means, stds)
