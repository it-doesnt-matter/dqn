from typing import SupportsFloat

import torch
from torchvision import transforms as T
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.core import WrapperObsType, WrapperActType


class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int) -> None:
        super().__init__(env)
        self.skip = skip

    def step(
        self,
        action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, any]]:
        total_reward = 0.0
        for _ in range(self.skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation
