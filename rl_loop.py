from torchrl import specs
import time
from tqdm import tqdm


def prefix_dict(prefix: str, d: dict) -> dict:
    """Prefixes all keys in `d` with `prefix`."""
    return {f"{prefix}/{k}": v for k, v in d.items()}


def train_loop(
    experiment,
    env_fn,
    agent_fn,
    replay_fn,
    max_steps: int,
    warmstart_steps: int,
    log_interval: int,
    checkpoint_interval: int,
    tqdm_bar: bool,
) -> None:
    env = env_fn()
    agent = agent_fn(env)
    replay_buffer = replay_fn(env)

    spec = specs.EnvironmentSpec.make(env)
    timestep = env.reset()
    replay_buffer.insert(timestep, None)

    start_time = time.time()
    for i in tqdm(range(1, max_steps + 1), disable=not tqdm_bar, dynamic_ncols=True):
        if i < warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent.eval()
            action = agent.sample_actions(timestep.observation)
            agent.train()

        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        if timestep.last():
            experiment.log(prefix_dict("train", env.get_statistics()), step=i)
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        if i >= warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                metrics = agent.update(transitions)
                if i % log_interval == 0:
                    experiment.log(prefix_dict("train", metrics), step=i)

        if checkpoint_interval >= 0 and i % checkpoint_interval == 0:
            experiment.save_checkpoint(agent, step=i)

        if i % log_interval == 0:
            experiment.log({"train/fps": int(i / (time.time() - start_time))}, step=i)
