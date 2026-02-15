"""Template for FMU-based SimEnv environments.

This template is used by SimEnv.from_fmu() to generate environment classes.
Variables in braces (e.g., {class_name}, {fmu_name}) are replaced during generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eta_ctrl.envs.sim_env import SimEnv

if TYPE_CHECKING:
    from typing import Any


class TemplateSimEnv(SimEnv):
    """Environment for TEMPLATE_FMU_NAME FMU simulation."""

    @property
    def fmu_name(self) -> str:
        """Name of the FMU file."""
        return "TEMPLATE_FMU_NAME"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TemplateSimEnv environment."""
        super().__init__(*args, **kwargs)

    def _step(self) -> tuple[float, bool, bool, dict]:
        """Perform one time step and return its results.

        This method should be customized for your specific FMU environment.
        The default implementation calls the parent SimEnv step method.

        :param action: Actions to perform in the environment.
        :return: Tuple of (observations, reward, terminated, truncated, info).
        """
        # Insert your custom step logic here
        # For example: custom action preprocessing

        # SimEnv populates the state with observations from the simulator
        _, terminated, truncated, info = super()._step()

        # Implement reward calculation
        reward = 0

        return reward, terminated, truncated, info

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the environment to an initial internal state.

        This method should be customized for your specific FMU environment.
        The default implementation calls the parent SimEnv reset method.

        :param seed: Random seed for environment initialization.
        :param options: Additional options for reset.
        :return: Tuple of (initial_observation, info).
        """
        # Insert your custom reset logic here
        # For example:
        # - Custom initial state setup
        # - Custom parameter initialization
        # - Custom observation preprocessing
        infos: dict[str, str] = {}

        # Call reset logic from SimEnv
        # This loads initial simulator observations into the state
        super()._reset()

        return infos

    # Add additional custom methods here:
    # - Custom reward functions
    # - Specialized observation processing
    # - Environment-specific state management
