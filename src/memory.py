import random
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.size = 0
        self.next_index = 0

    def total_sum(self) -> float:
        return self.tree[0]

    def add(self, priority: float, transition: Transition) -> None:
        self.data[self.next_index] = transition
        self.update(self.next_index, priority)

        self.next_index = (self.next_index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def update(self, index: int, priority: float) -> None:
        tree_index = index + self.capacity - 1
        change = priority - self.tree[tree_index]

        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _propagate(self, tree_index: int, change: float) -> None:
        parent_index = (tree_index - 1) // 2
        self.tree[parent_index] += change

        if parent_index != 0:
            self._propagate(parent_index, change)

    def get_by_prefix_sum(self, prefix_sum: float) -> tuple[int, float, Transition]:
        tree_index = self._get_prefix_sum_index(prefix_sum)
        index = tree_index - self.capacity + 1

        return (index, self.tree[tree_index], self.data[index])

    def _get_prefix_sum_index(self, prefix_sum: float) -> float:
        tree_index = 0
        while tree_index < self.capacity - 1:
            # if (sum of the left child) > prefix_sum: go to the left branch
            if self.tree[tree_index * 2 + 1] > prefix_sum:
                tree_index = tree_index * 2 + 1
            # else go to the right branch and take sum of the left child into account
            else:
                prefix_sum -= self.tree[tree_index * 2 + 1]
                tree_index = 2 * tree_index + 2
        return tree_index


class PrioritizedReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.max_priority = 1.0

        self.epsilon = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.0002

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + self.epsilon) ** self.alpha

    # new transitions come without a td_error
    # they get max_priority to guarantee that all transitions
    # are seen at least once
    def add_transition(self, transition: Transition) -> None:
        priority = self._get_priority(self.max_priority)
        self.tree.add(priority, transition)

    def sample_batch(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        batch = []
        indexes = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)
        segment_size = self.tree.total_sum() / batch_size

        self.beta += self.beta_increment
        self.beta = min(self.beta, 1)

        for i in range(batch_size):
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)

            prefix_sum = random.uniform(segment_start, segment_end)
            index, priority, transition = self.tree.get_by_prefix_sum(prefix_sum)

            batch.append(transition)
            indexes[i] = index
            priorities[i] = priority

        probabilities = priorities / self.tree.total_sum()
        is_weights = np.power(probabilities * self.tree.size, -self.beta)
        is_weights /= is_weights.max()

        return (batch, indexes, is_weights)

    def update_priorities(self, indexes: torch.Tensor, errors: torch.Tensor) -> None:
        for index, error in zip(indexes, errors):
            error = error.item()
            self.max_priority = max(self.max_priority, error)
            priority = self._get_priority(error)
            self.tree.update(index, priority)
