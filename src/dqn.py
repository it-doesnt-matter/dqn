from datetime import datetime
from pytz import timezone
from collections import deque
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optm
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers.frame_stack import LazyFrames
from tqdm import trange

from network import ConvQNetwork
from logger import SummaryLogger
from wrapper import SkipFrame, GrayScaleObservation, ResizeObservation


class DQNAgent:
    def __init__(
        self,
        double: bool,
        state_dim: list[int],
        n_actions: int,
        env: gym.Env | str,
        skip_frame: int,
        replay_capacity: int,
        replay_start_size: int,
        gamma: float,
        tau: float,
        epsilon_start: float,
        epsilon_decay: float,
        epsilon_min: float,
        batch_size: int,
        update_target_every: int,  # steps
        optimizer: type[optm.Optimizer],
        loss: type[nn.modules.loss._Loss],
        save_every: Optional[int] = None,  # episodes
        model_save_folder: str = "../saved_models",
        summary_save_folder: str = "../summaries",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.double = double
        self.current_step = 0

        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env
        self.env = SkipFrame(self.env, skip=skip_frame)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=state_dim[2])

        self.n_actions = n_actions
        self.state_dim = state_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = ConvQNetwork(state_dim, n_actions).to(self.device)
        self.target_net = ConvQNetwork(state_dim, n_actions).to(self.device)

        self.optimizer = optimizer(self.policy_net.parameters())
        self.loss = loss

        self.replay_memory = deque(maxlen=replay_capacity)
        self.replay_start_size = replay_start_size

        self.save_every = save_every
        self.model_save_folder = model_save_folder
        self.logger = SummaryLogger(summary_save_folder)

    def select_action(self, state: LazyFrames) -> int:
        rng = np.random.default_rng()

        if rng.random() < self.epsilon:
            action_index = rng.integers(self.n_actions)
        else:
            state = np.asarray(state)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.policy_net(state)
            action_index = torch.argmax(action_values, axis=1).item()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return action_index

    def save_experience(
        self,
        state: LazyFrames,
        action: int,
        reward: int,
        next_state: LazyFrames,
        done: bool,
    ) -> None:
        state = np.asarray(state)
        next_state = np.asarray(next_state)

        state = torch.tensor(state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        done = torch.tensor([done], device=self.device)

        self.replay_memory.append((state, action, reward, next_state, done))

    def sample_experiences(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.replay_memory, self.batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return (state, action.squeeze(), reward.squeeze(), next_state, done.squeeze())

    def calculate_td_estimates(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_index = np.arange(0, self.batch_size)
        predictions = self.policy_net(state)
        return predictions[batch_index, action]

    @torch.no_grad()
    def calculate_td_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        if self.double:
            policy_predictions = self.policy_net(next_states)
            best_actions = torch.argmax(policy_predictions, axis=1)
            batch_indexes = np.arange(0, self.batch_size)
            target_predictions = self.target_net(next_states)
            next_Qs = target_predictions[batch_indexes, best_actions]
        else:
            target_predictions = self.target_net(next_states)
            next_Qs, _ = torch.max(target_predictions, dim=1)

        return (rewards + (1 - dones.float()) * self.gamma * next_Qs).float()

    def update_policy_net(
        self,
        td_estimates: torch.Tensor,
        td_targets: torch.Tensor,
    ) -> float:
        loss = self.loss(td_estimates, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self) -> tuple[float, float]:  # TODO check if return type is correct
        if len(self.replay_memory) < self.replay_start_size:
            return (0.0, 0.0)

        states, actions, rewards, next_states, dones = self.sample_experiences()
        td_estimates = self.calculate_td_estimates(states, actions)
        td_targets = self.calculate_td_targets(rewards, next_states, dones)
        loss = self.update_policy_net(td_estimates, td_targets)

        return (loss, td_estimates.mean().item())

    def train(self, episodes: int, save_summary: bool = False) -> None:
        for episode in trange(episodes, unit="ep"):
            state, _ = self.env.reset()

            while True:
                self.current_step += 1

                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.save_experience(state, action, reward, next_state, done)
                loss, td_estimate = self.learn()

                self.logger.log_step(reward, loss, td_estimate)

                state = next_state

                if self.current_step % self.update_target_every == 0:
                    self.sync_target_net()

                if terminated or truncated:
                    break

            if (episode + 1) % self.save_every == 0:
                self.save_policy(f"epsiode_{episode + 1}")
            self.logger.log_episode(self.epsilon)

        self.logger.draw_training_summary(save_summary)

    def save_policy(self, file_name: Optional[str] = None) -> None:
        if file_name is None:
            local_tz = timezone("Europe/Luxembourg")
            local_datetime = datetime.now(local_tz)
            file_name = local_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"{self.model_save_folder}/{file_name}.pth"
        torch.save(self.policy_net.state_dict(), save_path)
