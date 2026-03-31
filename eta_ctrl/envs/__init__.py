from __future__ import annotations

from typing import TYPE_CHECKING

from .base_env import BaseEnv as BaseEnv
from .sim_env import SimEnv as SimEnv
from .state import (
    StateConfig as StateConfig,
    StateVar as StateVar,
)

if TYPE_CHECKING:
    from .live_env import LiveEnv as LiveEnv
    from .no_vec_env import NoVecEnv as NoVecEnv
    from .pyomo_env import PyomoEnv as PyomoEnv


def __getattr__(name: str) -> object:
    if name == "LiveEnv":
        from .live_env import LiveEnv  # noqa: PLC0415

        return LiveEnv
    if name == "NoVecEnv":
        from .no_vec_env import NoVecEnv  # noqa: PLC0415

        return NoVecEnv
    if name == "PyomoEnv":
        from .pyomo_env import PyomoEnv  # noqa: PLC0415

        return PyomoEnv
    msg = f"module 'eta_ctrl.envs' has no attribute {name!r}"
    raise AttributeError(msg)
