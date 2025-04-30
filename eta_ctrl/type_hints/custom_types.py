from __future__ import annotations

import datetime
from os import PathLike
from typing import Literal

import numpy as np

# Other custom types:
Path = str | PathLike
Number = float | int | np.floating | np.signedinteger | np.unsignedinteger
TimeStep = int | float | datetime.timedelta

FillMethod = Literal["ffill" | "fillna" | "bfill" | "interpolate" | "asfreq"]
