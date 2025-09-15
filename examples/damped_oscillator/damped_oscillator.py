from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eta_ctrl.common import episode_results_path
from eta_ctrl.envs import SimEnv

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from typing import Any

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.util.type_annotations import TimeStep


class DampedOscillatorEnv(SimEnv):
    """
    Damped oscillator environment class from SimEnv.
    Model settings come from fmu file.

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: callback which should be called after each episode
    :param scenario_time_begin: Beginning time of the scenario
    :param scenario_time_end: Ending time of the scenario
    :param episode_duration: Duration of the episode in seconds
    :param sampling_time: Duration of a single time sample / time step in seconds
    :param scale_actions: Normalize the actions when using RL algorithms
    """

    # Set info
    version = "v0.1"
    description = "Damped oscillator"
    fmu_name = "damped_oscillator"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigRun,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        scale_actions: bool = False,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id,
            config_run,
            verbose,
            callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
            **kwargs,
        )
        self.scale_actions = scale_actions

        # Initialize the simulator
        self._init_simulator()

        #: Total reward over an episode
        self.episode_reward: float = 0.0

    def _step(self) -> tuple[float, bool, bool, dict]:
        """Perform one time step and return its results. Set random force and perform the simulation.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """
        random_obs = self.observation_space.sample()
        self.state["f"] = random_obs["f"]

        _, terminated, truncated, info = super()._step()
        self.episode_reward -= abs(self.state["s"][0])
        return self.episode_reward, terminated, truncated, info

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the model and return initial observations.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`) (default: None).
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)
        :return: Tuple of observation and info. Analogous to the ``info`` returned by :meth:`step`.
        """

        random_obs = self.observation_space.sample()
        self.state["f"] = random_obs["f"]

        super()._reset()
        return {}

    def render(self, mode: str = "human") -> None:
        self.export_state_log(
            path=episode_results_path(self.config_run.path_series_results, self.run_name, 1, self.env_id)
        )

        mpl.rcParams["font.family"] = "Times New Roman"
        mpl.rcParams["font.size"] = "9"
        linestyles = [":", "--", "-"]

        def greys(x: int) -> tuple[float, ...]:
            return (*tuple([(x / 4) for _ in range(3)]), 1)

        fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
        fig.set_layout_engine("tight")

        # Convert from numpy arrays to scalar values
        scalar_data = [{k: v.item() for k, v in state.items() if v.size == 1} for state in self.state_log]
        data = pd.DataFrame(data=scalar_data, index=list(range(len(self.state_log))), dtype=np.float32)
        x = data.index
        columns = {"distance of mass": "s", "input signal": "u"}

        lines: list[mpl.lines.Line2D] = []
        labels: list[str] = []
        for name, col in columns.items():
            hdl = ax.plot(x, data[col], color=greys(len(lines)), linestyle=linestyles[len(lines)])[0]
            lines.append(hdl)
            labels.append(name)

        ax.legend(lines, labels, loc="upper right")
        ax.yaxis.grid(color="gray", linestyle="dashed")

        ax.set_xlabel("time")
        ax.set_ylabel("distance")

        plt.show()
