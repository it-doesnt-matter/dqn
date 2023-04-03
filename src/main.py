from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optm
import gymnasium as gym
from gymnasium.utils.play import play
from gymnasium.wrappers import FrameStack

from network import ConvQNetwork, DuelingQNetwork
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation
from dqn import DQNAgent
from pre_dqn import PREDQNAgent


Algorithm = Enum("Algorithm", ["DEFAULT", "DOUBLE", "PRIORITIZED", "DUELING"])


def predict(file_path: str, algo: Algorithm) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo in [Algorithm.DEFAULT, Algorithm.DOUBLE, Algorithm.PRIORITIZED]:
        model = ConvQNetwork([84, 84, 4], 5).to(device)
    elif algo is Algorithm.DUELING:
        model = DuelingQNetwork([84, 84, 4], 5).to(device)
    else:
        return
    model.load_state_dict(torch.load(file_path))

    env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    state, _ = env.reset()

    while True:
        state = np.asarray(state)
        state = torch.tensor(state, device=device).unsqueeze(0)
        with torch.inference_mode():
            action_values = model(state)
        action = torch.argmax(action_values).item()
        state, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()


def train(n_episodes: int, algo: Algorithm) -> None:
    env = gym.make("CarRacing-v2", continuous=False)

    kwargs = {
        "state_dim": [84, 84, 4],
        "n_actions": 5,
        "env": env,
        "skip_frame": 4,
        "replay_capacity": 100_000,
        "replay_start_size": 10_000,
        "gamma": 0.99,
        "tau": 0.005,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.999975,
        "epsilon_min": 0.1,
        "batch_size": 64,
        "update_target_every": 5_000,
        "optimizer": optm.Adam,
        "save_every": 50,
    }

    if algo is Algorithm.DEFAULT:
        kwargs["double"] = False
        kwargs["loss"] = nn.HuberLoss()
        agent = DQNAgent(**kwargs)
    elif algo is Algorithm.DOUBLE:
        kwargs["double"] = True
        kwargs["loss"] = nn.HuberLoss()
        agent = DQNAgent(**kwargs)
    elif algo is Algorithm.PRIORITIZED:
        kwargs["double"] = True
        kwargs["dueling"] = False
        kwargs["loss"] = nn.HuberLoss(reduction="none")
        agent = PREDQNAgent(**kwargs)
    elif algo is Algorithm.DUELING:
        kwargs["double"] = True
        kwargs["dueling"] = True
        kwargs["loss"] = nn.HuberLoss(reduction="none")
        agent = PREDQNAgent(**kwargs)
    else:
        return

    agent.train(n_episodes, save_summary=True)
    agent.save_policy()


def manual_play() -> None:
    play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={
                                               "w": np.array([0, 0.7, 0]),
                                               "a": np.array([-1, 0, 0]),
                                               "s": np.array([0, 0, 1]),
                                               "d": np.array([1, 0, 0]),
                                               "wa": np.array([-1, 0.7, 0]),
                                               "dw": np.array([1, 0.7, 0]),
                                               "ds": np.array([1, 0, 1]),
                                               "as": np.array([-1, 0, 1]),
                                              }, noop=np.array([0, 0, 0]))


if __name__ == "__main__":
    train(5_000, Algorithm.DUELING)
