from __future__ import annotations

import datetime
from os import PathLike
from typing import Any, Literal

import numpy as np
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymEnv,
    GymObs as ObservationType,
    GymResetReturn as ResetResult,
    GymStepReturn as StepResult,
    MaybeCallback,
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
