from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

try:
    import pygame  # noqa: F401
except ModuleNotFoundError:
    msg = (
        "For the PendulumEnv example, the pygame module is required. Install eta_ctrl with the "
        "[examples] option to get all packages required for running examples."
    )
    raise ModuleNotFoundError(
        msg,
        name="pygame",
    ) from None
else:
    from gymnasium.envs.classic_control.pendulum import (
        PendulumEnv as GymPendulum,
        angle_normalize,
    )

from eta_ctrl.envs import BaseEnv, StateConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from typing import Any

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.util.type_annotations import TimeStep

log = getLogger(__name__)


class PendulumEnv(BaseEnv, GymPendulum):
    """Pendulum environment in the *eta_ctrl* style. This class is adapted from the
    `pendulum example <https://gymnasium.farama.org/environments/classic_control/pendulum/>`_ in
    the Farama gymnasium: :py:class:`gymnasium.envs.classic_control.PendulumEnv`:

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: callback which should be called after each episode
    :param scenario_time_begin: Beginning time of the scenario
    :param scenario_time_end: Ending time of the scenario
    :param episode_duration: Duration of the episode in seconds
    :param sampling_time: Duration of a single time sample / time step in seconds
    :param max_speed: Maximum speed of the pendulum
    :param max_torque: Maximum torque that can be applied by the agent
    :param g: Gravitational acceleration
    :param mass: Mass at the tip of the pendulum
    :param length: Length of the pendulum
    :param do_render: Render the learning/playing process or not? (default: True)
    :param screen_dim: Dimension of the screen to render on in pixels (default: 500)
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    """

    version = "v1.0"
    description = "OpenAI"

    def __init__(  # noqa: PLR0913
        self,
        env_id: int,
        config_run: ConfigRun,
        verbose: int = 2,
        callback: Callable | None = None,
        seed: int | None = None,
        *,
        state_config: StateConfig,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        max_speed: int,
        max_torque: float,
        g: float,
        mass: float,
        length: float,
        do_render: bool = True,
        screen_dim: int = 500,
        render_mode: str = "human",
    ) -> None:
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            state_config=state_config,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
        )

        # Load environment dynamics specific settings
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.g = g
        self.mass = mass
        self.length = length

        # Other
        self.screen_dim = screen_dim
        self.do_render = do_render

        # Initialize counters
        self.last_u: float | None = None

    def _step(self) -> tuple[float, bool, bool, dict]:
        """See base_env documentation"""

        # Get previous step values (th := theta)
        th = self.state_log[-1]["th"]
        thdot = self.state_log[-1]["th_dot"]

        # Clip input from agent by max values
        self.state["torque"] = np.clip(self.state["torque"], -self.max_torque, self.max_torque)
        self.last_u = self.state["torque"][0]  # for rendering

        # Calculate state of the pendulum
        # Constants
        g, m, length, dt = self.g, self.mass, self.length, self.sampling_time
        # Variables
        u = self.state["torque"]
        newthdot = thdot + (3 * g / (2 * length) * np.sin(th) + 3.0 / (m * length**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state["th_dot"] = newthdot
        self.state["th"] = newth
        self.state["cos_th"] = np.cos(self.state["th"])
        self.state["sin_th"] = np.sin(self.state["th"])

        # reward function
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        # Check if the episode is completed
        terminated = self._terminated()
        truncated = False

        return -costs[0], terminated, truncated, {}

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`) (default: None).
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)
        :return: Tuple of observation and info. Analogous to the ``info`` returned by :meth:`step`.
        """

        # save episode's stats
        self.last_u = None
        random_obs = self.observation_space.sample()
        self.state["th"] = random_obs["th"]
        self.state["th_dot"] = random_obs["th_dot"]
        # Calculate sin and cos
        self.state["cos_th"] = np.cos(self.state["th"])
        self.state["sin_th"] = np.sin(self.state["th"])

        return {}

    def render(self) -> None:
        """Use the render function from the Farama gymnasium PendulumEnv environment.
        This requires a little hack because our 'self.state' attribute is different."""
        if self.do_render:
            state = self.state.copy()
            self.state = (
                [self.state["th"], self.state["th_dot"]] if not isinstance(state, np.ndarray) else state  # type: ignore[unreachable, assignment]
            )
            GymPendulum.last_u = self.last_u
            GymPendulum.render(self)
            self.state = state

    def close(self) -> None:
        GymPendulum.close(self)
