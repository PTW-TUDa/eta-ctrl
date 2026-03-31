from __future__ import annotations

from typing import TYPE_CHECKING

from .fmu import (
    FMU2MESlave as FMU2MESlave,
    FMUSimulator as FMUSimulator,
)

if TYPE_CHECKING:
    from .pyomo_model import PyomoModel as PyomoModel


def __getattr__(name: str) -> object:
    if name == "PyomoModel":
        from .pyomo_model import PyomoModel  # noqa: PLC0415

        return PyomoModel
    msg = f"module 'eta_ctrl.simulators' has no attribute {name!r}"
    raise AttributeError(msg)
