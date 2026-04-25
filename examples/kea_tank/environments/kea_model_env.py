from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from eta_ctrl.envs.base_env import BaseEnv

if TYPE_CHECKING:
    from typing import Any


log = getLogger(__name__)


class DrKea(BaseEnv):
    version = "0.1"
    description = "Simple environment for kea example."

    def _step(self) -> tuple[float, bool, bool, dict]:
        if self.state["heating"] == 1:
            self.temp += 0.02 * self.sampling_time
        else:
            self.temp -= 0.01 * self.sampling_time

        self.state["tank_temperature_start"] = np.array([self.temp])
        return 0, False, self._truncated(), {}

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.temp = 60.0
        self.state["tank_temperature_start"] = np.array([self.temp])

        return {}

    def close(self) -> None:
        pass

    def render(self) -> None:
        pass
