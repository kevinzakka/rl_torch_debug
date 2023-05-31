from typing import Optional, Sequence
from dataclasses import dataclass
from torchrl.specs import EnvironmentSpec
from torchrl.types import Transition
from torchrl import networks, distributions
import numpy as np
from functools import partial
import torch
import torch.nn as nn


@dataclass(frozen=True)
class SACConfig:
    """Configuration options for SAC."""

    num_qs: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256)
    critic_dropout_rate: float = 0.0
    critic_layer_norm: bool = False
    tau: float = 0.005
    target_entropy: Optional[float] = None
    init_temperature: float = 1.0
    backup_entropy: bool = True


class SAC:
    def __init__(
        self,
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        action_dim = spec.action.shape[-1]
        obs_dim = spec.observation.shape[-1]

        actor_base_cls = partial(
            networks.MLP,
            input_dim=obs_dim,
            hidden_dims=config.hidden_dims,
            activation=nn.ReLU,
            activate_final=True,
        )
        actor = distributions.TanhNormal(
            base_cls=actor_base_cls,
            base_cls_output_dim=config.hidden_dims[-1],
            action_dim=action_dim,
            squash_tanh=True,
        )

        critic_base_cls = partial(
            networks.MLP,
            input_dim=obs_dim + action_dim,
            hidden_dims=config.hidden_dims,
            activation=nn.ReLU,
            activate_final=True,
            use_layer_norm=config.critic_layer_norm,
            dropout_rate=config.critic_dropout_rate,
        )
        critic_cls = partial(
            networks.StateActionValue,
            base_cls=critic_base_cls,
            base_cls_output_dim=config.hidden_dims[-1],
        )
        critic = networks.Ensemble(critic_cls, num=config.num_qs)
        target_critic = networks.Ensemble(critic_cls, num=config.num_qs)
        target_critic.load_state_dict(critic.state_dict())

        temperature = nn.Parameter(
            torch.tensor(config.init_temperature, dtype=torch.float32).log(),
            requires_grad=True,
        )

        actor.to(device)
        critic.to(device)
        target_critic.to(device)
        temperature.to(device)

        actor = torch.compile(actor)
        critic = torch.compile(critic)
        target_critic = torch.compile(target_critic)
        temperature = torch.compile(temperature)

        self._actor = actor
        self._critic = critic
        self._target_critic = target_critic
        self._temperature = temperature

        self._actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self._critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config.critic_lr
        )
        self._temperature_optimizer = torch.optim.Adam([temperature], lr=config.temp_lr)

        self._target_entropy = config.target_entropy or -0.5 * action_dim
        self._seed = seed
        self._discount = discount
        self._backup_entropy = config.backup_entropy
        self._tau = config.tau
        self._device = device

    def update(self, transitions: Transition) -> dict:
        critic_info = self._update_critic(transitions)
        actor_info = self._update_actor(transitions)
        temp_info = self._update_temperature(actor_info["entropy"])
        return {**critic_info, **actor_info, **temp_info}

    @torch.no_grad()
    def sample_actions(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(observation).float().to(self._device)
        obs_tensor.unsqueeze_(0)
        return self._actor(obs_tensor).sample().squeeze(0).cpu().numpy()

    @torch.no_grad()
    def eval_actions(self, observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(observation).float().to(self._device)
        obs_tensor.unsqueeze_(0)
        return self._actor(obs_tensor).mode().squeeze(0).cpu().numpy()

    def eval(self):
        self._actor.eval()

    def train(self):
        self._actor.train()

    # Private API.

    def _update_critic(self, transitions: Transition) -> dict:
        next_dist = self._actor(transitions.next_observation)
        next_actions = next_dist.sample()
        next_log_probs = next_dist.log_prob(next_actions).sum(-1)

        q1, q2 = self._target_critic(transitions.next_observation, next_actions)
        next_q = torch.minimum(q1, q2)
        target_q = transitions.reward + self._discount * transitions.discount * next_q

        if self._backup_entropy:
            target_q = target_q - (
                self._discount
                * transitions.discount
                * self._temperature.exp().detach()
                * next_log_probs
            )

        qs = self._critic(transitions.observation, transitions.action)
        self._critic_optimizer.zero_grad(True)
        critic_loss = (qs - target_q).pow(2).mean()
        critic_loss.backward()
        self._critic_optimizer.step()

        with torch.no_grad():
            for t, s in zip(
                self._target_critic.parameters(), self._critic.parameters()
            ):
                t.data.copy_(self._tau * s.data + (1 - self._tau) * t.data)

        return {"critic_loss": critic_loss.item(), "q": qs.mean().item()}

    def _update_actor(self, transitions: Transition) -> dict:
        dist = self._actor(transitions.observation)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(-1)
        qs = self._critic(transitions.observation, actions)
        q = torch.mean(qs, dim=0)
        self._actor_optimizer.zero_grad(True)
        actor_loss = (log_probs * self._temperature.exp().detach() - q).mean()
        actor_loss.backward()
        self._actor_optimizer.step()
        return {
            "actor_loss": actor_loss.item(),
            "entropy": -1 * log_probs.mean().item(),
        }

    def _update_temperature(self, entropy: float) -> dict:
        self._temperature_optimizer.zero_grad(True)
        temp = self._temperature.exp()
        temp_loss = (temp * (entropy - self._target_entropy)).mean()
        temp_loss.backward()
        self._temperature_optimizer.step()
        return {
            "temperature_loss": temp_loss.item(),
            "temperature": temp.item(),
        }
