from typing import Optional, Tuple
from dataclasses import dataclass, asdict
import tyro
import time
from pathlib import Path

import torch
import numpy as np
import random

from torchrl import sac
from torchrl import specs
from torchrl import replay

from rl_loop import train_loop
from rl_experiment import Experiment
from dm_control import suite
import dm_env_wrappers as wrappers

_PROJECT = "torchrl"
_ROOT_DIR = "/tmp/torchrl/runs"


@dataclass(frozen=True)
class Args:
    seed: int = 0
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    checkpoint_interval: int = -1
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    use_wandb: bool = False
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    clip: bool = False
    record_every: int = 1
    record_resolution: Tuple[int, int] = (240, 320)
    camera_id: Optional[str | int] = 0
    agent_config: sac.SACConfig = sac.SACConfig()
    domain_name: str = "cartpole"
    task_name: str = "swingup"


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.domain_name}-{args.task_name}-{args.seed}-{time.time()}"

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setup the experiment for checkpoints, videos, metadata, etc.
    experiment = Experiment(Path(_ROOT_DIR) / run_name).assert_new()
    experiment.write_metadata("config", args)

    if args.use_wandb:
        experiment.enable_wandb(
            project=_PROJECT,
            entity=args.entity or None,
            tags=(args.tags.split(",") if args.tags else []),
            notes=args.notes or None,
            config=asdict(args),
            mode=args.mode,
            name=run_name,
            sync_tensorboard=True,
        )

    def agent_fn(env) -> sac.SAC:
        return sac.SAC(
            spec=specs.EnvironmentSpec.make(env),
            config=args.agent_config,
            seed=args.seed,
            discount=args.discount,
            device=device,
        )

    def replay_fn(env) -> replay.ReplayBuffer:
        return replay.ReplayBuffer(
            capacity=args.replay_capacity,
            batch_size=args.batch_size,
            spec=specs.EnvironmentSpec.make(env),
            device=device,
        )

    def env_fn():
        env = suite.load(
            args.domain_name, args.task_name, task_kwargs={"random": args.seed}
        )
        env = wrappers.EpisodeStatisticsWrapper(env, deque_size=1)
        env = wrappers.ConcatObservationWrapper(env)
        env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.DmControlWrapper(env)
        return env

    train_loop(
        experiment=experiment,
        env_fn=env_fn,
        agent_fn=agent_fn,
        replay_fn=replay_fn,
        max_steps=args.max_steps,
        warmstart_steps=args.warmstart_steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        tqdm_bar=args.tqdm_bar,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
