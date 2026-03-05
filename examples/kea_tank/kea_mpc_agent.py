from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from eta_ctrl.agents.mpc_agent import MpcAgent

if TYPE_CHECKING:
    from typing import Any

    import numpy as np


log = getLogger(__name__)


class KeaMpcAgent(MpcAgent):
    model_file = Path("../kea_pyomo_model.py")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Plot the solution after solving."""
        action_array, state = super().predict(observation=observation)

        return action_array, state
