import torch
import torch.nn as nn


class BasicQNetwork(nn.Module):

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.net(input_)


class ConvQNetwork(nn.Module):
    def __init__(self, n_inputs: list[int], n_outputs: int) -> None:
        super().__init__()
        height, width, n_frames = n_inputs
        self.net = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_outputs),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.net(input_)


class DuelingQNetwork(nn.Module):
    def __init__(self, n_inputs: list[int], n_outputs: int) -> None:
        super().__init__()
        height, width, n_frames = n_inputs
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        conv_result = self.conv(input_)

        value_result = self.value_stream(conv_result)
        advantage_result = self.advantage_stream(conv_result)

        return value_result + advantage_result - advantage_result.mean(dim=-1, keepdim=True)
