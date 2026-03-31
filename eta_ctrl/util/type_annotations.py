from __future__ import annotations

import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3.common.type_aliases import (
        GymEnv,  # noqa: F401
        GymObs as ObservationType,  # noqa: F401
        GymResetReturn as ResetResult,  # noqa: F401
        GymStepReturn as StepResult,  # noqa: F401
        MaybeCallback,  # noqa: F401
    )

# Other custom types:
Path = str | PathLike
Number = float | int | np.floating | np.signedinteger | np.unsignedinteger
TimeStep = int | float | datetime.timedelta

FillMethod = Literal["ffill", "bfill", "interpolate", "asfreq"]
InferDatetimeType = Literal["string", "dates"]

ActionType = np.ndarray
EnvSettings = dict[str, Any]
AlgoSettings = dict[str, Any]
PyoParams = dict[str | None, dict[str | None, Any] | Any]
