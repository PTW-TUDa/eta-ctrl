from typing import Any

from eta_ctrl.envs import PyomoSimEnv


class DrKeaPyomoSimEnv(PyomoSimEnv):
    version = "0.1.0"
    description = "PyomoSimEnv that simulates the DrKea tank"

    @property
    def model_import(self) -> str:
        return "examples.kea_tank.kea_pyomo_model.DrKeaModel"

    def render(self, **kwargs: Any) -> None: ...
