from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyomo import environ as pyo

from eta_ctrl.util.utils import is_divisible

if TYPE_CHECKING:
    from typing import Any

    from eta_ctrl.util.type_annotations import PyoParams, TimeStep


log = getLogger(__name__)


class PyomoModel:
    def __init__(
        self,
        *,
        sampling_time: float,
        model_parameters: dict[str, Any] | None = None,
        prediction_horizon: TimeStep | str | None = None,
        **kwargs: Any,
    ) -> None:
        #: Sampling time (interval between optimization time steps) in seconds.
        self.sampling_time = sampling_time

        # #: Total duration of one prediction/optimization run when used with the MPC agent.
        if prediction_horizon is None:
            log.error("Prediction_horizon parameter is not present in config.")
            raise ValueError
        self.prediction_horizon = float(
            prediction_horizon if not isinstance(prediction_horizon, timedelta) else prediction_horizon.total_seconds()
        )

        if not is_divisible(self.prediction_horizon, self.sampling_time):
            msg = (
                "The sampling_time must fit evenly into the prediction_horizon "
                "(prediction_horizon % sampling_time must equal 0)."
            )
            raise ValueError(msg)

        #: Number of steps in the prediction (prediction_horizon/sampling_time).
        self.n_prediction_steps: int = int(self.prediction_horizon / self.sampling_time)

        #: Configuration for the MILP model parameters.
        self.model_parameters = (model_parameters or {}).copy()  # prevent modifying original parameters

        abstract_model = self._model()
        #: Concrete pyomo model as initialized by _model.
        self.model: pyo.ConcreteModel = abstract_model.create_instance(data=self._pyo_init_params())

    @abc.abstractmethod
    def _model(self) -> pyo.AbstractModel:
        """Create the abstract pyomo model. This is where the pyomo model description should be placed.

        :return: Abstract pyomo model.
        """
        msg = "The abstract MPC environment does not implement a model."
        raise NotImplementedError(msg)

    def pyo_update_params(
        self,
        updated_params: dict[str, int | float | bool | Mapping | np.ndarray | Sequence | Any],
    ) -> None:
        """Update model parameters and indexed parameters of a pyomo instance with values given in a dictionary.

        :param updated_params: Dictionary with the updated values.
        :return: Updated model instance.
        """

        def update_scalar_component(component: pyo.Component, new_value: Any) -> None:
            if isinstance(new_value, (np.ndarray, Sequence)) and len(new_value) == 1:
                new_value = float(new_value[0])
            if not isinstance(new_value, (int, float, bool)):
                msg = f"Received non-scalar value {new_value} for component '{component_name}'"
                raise TypeError(msg)
            component.value = new_value

        def update_indexed_component(component: pyo.Component, new_values: Any) -> None:
            if isinstance(new_values, (Sequence, np.ndarray, pd.Series, Mapping)):
                len_ = len(new_values)
                if len_ == 1:
                    new_values = float(new_values[0])
                elif len_ != len(component):
                    msg = f"Component '{component}' needs {len(component)} values but {len_} were supplied."
                    raise ValueError(msg)

            if isinstance(new_values, (int, float, bool)):
                log.debug(f"Received a scalar value for indexed component '{component}', setting the first value")
                component[next(iter(component))] = new_values
                return

            if isinstance(new_values, Mapping):
                for param_val in list(component):
                    component[param_val] = new_values[param_val]
                return

            if isinstance(new_values, (Sequence, np.ndarray, pd.Series)):
                for i, param_val in enumerate(list(component)):
                    component[param_val] = float(new_values[i])
                return
            msg = f"Received unsupported datatype {type(new_values)} for component '{component}'"
            raise TypeError(msg)

        for component in self.model.component_objects():
            component_name = str(component)
            if component_name not in updated_params:
                # last entry is the parameter name for abstract models which are instanced
                component_name = component_name.rsplit(".", maxsplit=1)[-1]
                if component_name not in updated_params:
                    continue

            param_value = updated_params[component_name]
            # update simple components (single values)
            if not component.is_indexed():
                update_scalar_component(component=component, new_value=param_value)
            # update indexed components (time series)
            else:
                update_indexed_component(component=component, new_values=param_value)

        log.debug("Pyomo model parameters updated.")

    def _pyo_init_params(self) -> PyoParams:
        """Retrieve initial pyomo model parameters.

        Uses the values supplied by model_parameters.

        :return: Pyomo parameter dictionary.
        """
        if not self.model_parameters:
            return {}

        params = self.model_parameters.copy()
        out_raw = {name: (float(value) if value in ("inf", "-inf") else value) for name, value in params.items()}
        # Create mappings for pyomo
        out: PyoParams = {name: {None: value} for name, value in out_raw.items()}

        return {None: out}

    def pyo_get_solution(self, names: set[str] | None = None) -> dict[str, float | list[float]]:
        """Convert the pyomo solution into a more usable format for plotting.

        :param names: Names of the model parameters that are returned.
        :return: Dictionary of {parameter name: value} pairs. Value may be a scalar value or a list.
        """
        solution = {}
        for com in self.model.component_objects():
            if com.ctype not in {pyo.Var, pyo.Param, pyo.Objective}:
                continue
            if names is not None and com.name not in names:
                continue  # Only include names that where asked for
            solution[com.name] = [pyo.value(v) for v in com.values()] if com.is_indexed() else pyo.value(com)
        return solution
