"""Template for FMU-based SimEnv environments.

This template is used by SimEnv.from_fmu() to generate environment classes.
Variables in braces (e.g., {class_name}, {fmu_name}) are replaced during generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eta_ctrl.envs.sim_env import SimEnv

if TYPE_CHECKING:
    from typing import Any

    import numpy as np

    from eta_ctrl.util.type_annotations import ObservationType, StepResult


class TemplateSimEnv(SimEnv):
    """Environment for TEMPLATE_FMU_NAME FMU simulation."""

    @property
    def fmu_name(self) -> str:
        """Name of the FMU file."""
        return "TEMPLATE_FMU_NAME"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the TemplateSimEnv environment."""
        super().__init__(*args, **kwargs)

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        This method should be customized for your specific FMU environment.
        The default implementation calls the parent SimEnv step method.

        :param action: Actions to perform in the environment.
        :return: Tuple of (observations, reward, terminated, truncated, info).
        """
        # Insert your custom step logic here
        # For example:
        # - Custom action preprocessing
        # - Custom reward calculation
        # - Custom termination conditions

        # Default implementation uses parent class
        return super().step(action)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
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

        # Default implementation uses parent class
        return super().reset(seed=seed, options=options)

    # Add additional custom methods here:
    # - Custom reward functions
    # - Specialized observation processing
    # - Environment-specific state management
