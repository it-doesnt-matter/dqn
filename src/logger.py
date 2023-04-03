from datetime import datetime

from pytz import timezone
import matplotlib as mpl
import matplotlib.pyplot as plt


class SummaryLogger:
    def __init__(self, save_folder: str) -> None:
        self.save_folder = save_folder

        self.reset()

        self.episode_lengths: list[int] = []
        self.episode_rewards: list[int] = []
        self.episode_losses: list[float] = []
        self.episode_estimates: list[float] = []
        self.episode_epsilon: list[float] = []

    def reset(self) -> None:
        self.step_counter = 0
        self.step_rewards = 0
        self.step_losses = 0.0
        self.step_estimates = 0.0

    def log_step(self, reward: int, loss: float, estimate: float) -> None:
        self.step_counter += 1
        self.step_rewards += reward
        self.step_losses += loss
        self.step_estimates += estimate

    def log_episode(self, epsilon: float) -> None:
        self.episode_lengths.append(self.step_counter)
        self.episode_rewards.append(self.step_rewards)
        self.episode_losses.append(self.step_losses / self.step_counter)
        self.episode_estimates.append(self.step_estimates / self.step_counter)
        self.episode_epsilon.append(epsilon)

        self.reset()

    def draw_training_summary(self, save_summary: bool = False) -> None:
        self.set_mpl_theme()

        _, axs = plt.subplot_mosaic([
            ["length", "reward"],
            ["loss", "epsilon"],
        ])

        axs["length"].plot(self.episode_lengths)
        axs["length"].set_title("Episode Length")

        axs["reward"].plot(self.episode_rewards)
        axs["reward"].set_title("Episode Reward")

        axs["loss"].plot(self.episode_losses)
        axs["loss"].set_title("Episode Loss")

        axs["epsilon"].plot(self.episode_epsilon)
        axs["epsilon"].set_title("Epsilon")

        if save_summary:
            local_tz = timezone("Europe/Luxembourg")
            local_datetime = datetime.now(local_tz)
            datetime_str = local_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            save_path = f"{self.save_folder}/{datetime_str}.png"
            plt.savefig(save_path)
        plt.show()

    def set_mpl_theme(self) -> None:
        mpl.rcParams["patch.edgecolor"] = "w"
        mpl.rcParams["font.size"] = 10
        mpl.rcParams["patch.force_edgecolor"] = True
        mpl.rcParams["text.color"] = "0.15"
        mpl.rcParams["axes.facecolor"] = "#eaeaf2"
        mpl.rcParams["axes.edgecolor"] = "#ffffff"
        mpl.rcParams["axes.linewidth"] = 1.25
        mpl.rcParams["axes.grid"] = True
        mpl.rcParams["axes.titlesize"] = 12
        mpl.rcParams["axes.axisbelow"] = True
        mpl.rcParams["xtick.color"] = "0.15"
        mpl.rcParams["xtick.major.size"] = 6
        mpl.rcParams["xtick.minor.size"] = 4
        mpl.rcParams["xtick.major.width"] = 1.25
        mpl.rcParams["xtick.minor.width"] = 1
        mpl.rcParams["ytick.color"] = "0.15"
        mpl.rcParams["ytick.major.size"] = 6
        mpl.rcParams["ytick.minor.size"] = 4
        mpl.rcParams["ytick.major.width"] = 1.25
        mpl.rcParams["ytick.minor.width"] = 1
        mpl.rcParams["grid.color"] = "#ffffff"
        mpl.rcParams["grid.linewidth"] = 1
        mpl.rcParams["figure.figsize"] = (15, 5)
        mpl.rcParams["figure.constrained_layout.use"] = True
