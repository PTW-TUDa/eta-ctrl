from __future__ import annotations

import abc
import importlib
from collections.abc import Mapping, Sequence
from datetime import timedelta
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyomo import environ as pyo

from eta_ctrl.util.utils import is_divisible

if TYPE_CHECKING:
    import pathlib
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
            msg = "Prediction_horizon parameter is not present in config."
            raise ValueError(msg)
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
        updated_params: Mapping[str, int | float | bool | Mapping | np.ndarray | Sequence | Any],
    ) -> None:
        """Update model parameters and indexed parameters of a pyomo instance with values given in a dictionary.

        :param updated_params: Dictionary with the updated values.
        :return: Updated model instance.
        """

        def update_scalar_component(component: pyo.Param, new_value: Any) -> None:
            if isinstance(new_value, (np.ndarray, Sequence)) and len(new_value) == 1:
                new_value = float(new_value[0])
            if not isinstance(new_value, (int, float, bool)):
                msg = f"Received non-scalar value {new_value} for component '{component_name}'"
                raise TypeError(msg)
            component.value = new_value

        def update_indexed_component(component: pyo.Param, new_values: Any) -> None:
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

    def pyo_init_params(self) -> PyoParams:
        """Public proxy for initial parameter mapping used by utility workflows."""
        return self._pyo_init_params()

    def build_abstract_model(self) -> pyo.AbstractModel:
        """Public proxy for creating the model definition."""
        return self._model()

    @classmethod
    def _resolve_model_class(cls, model_import: str) -> type[PyomoModel]:
        """Resolve and return a PyomoModel subclass from a dotted import path."""
        module_path, cls_name = model_import.rsplit(".", 1)
        target_class = getattr(importlib.import_module(module_path), cls_name)
        if not issubclass(target_class, PyomoModel):
            msg = f"Imported class '{cls_name}' is not a subclass of PyomoModel."
            raise TypeError(msg)
        return target_class

    @classmethod
    def _create_export_probe(
        cls,
        target_class: type[PyomoModel],
        sampling_time: float,
        prediction_horizon: float,
        model_parameters: dict[str, Any] | None = None,
    ) -> PyomoModel:
        """Create a lightweight model probe for export without running subclass ``__init__``."""
        probe = target_class.__new__(target_class)
        probe.sampling_time = sampling_time
        probe.prediction_horizon = prediction_horizon
        probe.n_prediction_steps = int(prediction_horizon / sampling_time)
        probe.model_parameters = (model_parameters or {}).copy()

        return probe

    @classmethod
    def _create_concrete_model_for_export(
        cls,
        model_import: str,
        *,
        sampling_time: float,
        prediction_horizon: float,
        model_parameters: dict[str, Any] | None = None,
    ) -> pyo.ConcreteModel:
        """Build a concrete Pyomo model for export without requiring runtime setup parameters.

        This path intentionally bypasses subclass ``__init__`` to avoid side effects
        and parameter checks that are unrelated to static structure export.
        """
        if prediction_horizon <= 0 or sampling_time <= 0:
            msg = "sampling_time and prediction_horizon must be positive values."
            raise ValueError(msg)

        if not is_divisible(prediction_horizon, sampling_time):
            msg = (
                "The sampling_time must fit evenly into the prediction_horizon "
                "(prediction_horizon % sampling_time must equal 0)."
            )
            raise ValueError(msg)

        target_class = cls._resolve_model_class(model_import)
        probe = cls._create_export_probe(
            target_class,
            sampling_time=sampling_time,
            prediction_horizon=prediction_horizon,
            model_parameters=model_parameters,
        )

        abstract_model = probe.build_abstract_model()

        # Fill missing scalar parameters with neutral defaults to allow
        # concrete model instantiation for static export purposes.
        for component in abstract_model.component_objects(pyo.Param):
            if component.is_indexed():
                continue
            probe.model_parameters.setdefault(component.name, 0.0)

        return abstract_model.create_instance(data=probe.pyo_init_params())

    @classmethod
    def load_from_import(cls, model_import: str, **kwargs: Any) -> PyomoModel:
        """Load a :class:`PyomoModel` subclass from a dotted Python import string.

        This is the single place where model classes are resolved from their
        import path, so both :class:`~eta_ctrl.agents.MpcAgent` and
        :meth:`create_state` can reuse it without duplicating the logic.

        :param model_import: Dotted import path to the subclass
            (e.g. ``"eta_ctrl.examples.kea_tank.kea_pyomo_model.DrKeaModel"``).
        :param kwargs: Keyword arguments forwarded to the subclass constructor
            (e.g. ``sampling_time``, ``prediction_horizon``, ``model_parameters``).
        :return: Instantiated :class:`PyomoModel` subclass.
        """
        target_class = cls._resolve_model_class(model_import)
        return target_class(**kwargs)

    @classmethod
    def create_state(
        cls,
        model_import: str,
        model_name: str,
        output_dir: pathlib.Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Generate state config and model parameters TOML files for a PyomoModel.

        Creates a concrete model from the subclass ``_model`` definition (without
        running subclass ``__init__``), then writes:

        * ``{model_name}_state_config.toml`` — indexed ``pyo.Var`` components as
          actions and indexed ``pyo.Param`` components as observations.
        * ``{model_name}_model_parameters.toml`` — scalar ``pyo.Param`` components
          that belong in ``[agent_specific.model_parameters]`` of the run config.

        :param model_import: Dotted import path to the :class:`PyomoModel` subclass
            (e.g. ``"eta_ctrl.examples.kea_tank.kea_pyomo_model.DrKeaModel"``).
        :param model_name: Name used as prefix for the output files.
        :param output_dir: Target directory for the output files.
            Defaults to the current working directory.
        :param kwargs: Optional export-time kwargs:
            ``sampling_time`` (default ``1.0``), ``prediction_horizon``
            (default ``sampling_time``), and ``model_parameters``.
        """
        # Import here to avoid a circular dependency at module load time
        from eta_ctrl.common.export_pyomo import export_pyomo_model_state  # noqa: PLC0415

        sampling_time = float(kwargs.get("sampling_time", 1.0))
        prediction_horizon = float(kwargs.get("prediction_horizon", sampling_time))
        model_parameters = kwargs.get("model_parameters")

        concrete_model = cls._create_concrete_model_for_export(
            model_import,
            sampling_time=sampling_time,
            prediction_horizon=prediction_horizon,
            model_parameters=model_parameters,
        )

        export_pyomo_model_state(concrete_model, model_name, output_dir)

    def pyo_get_solution(self, names: set[str] | None = None) -> tuple[dict[str, list[float]], dict[str, float]]:
        """Convert the pyomo solution into a more usable format for plotting.

        :param names: Names of the model parameters that are returned.
        :return: Dictionary of {parameter name: value} pairs. Value may be a scalar value or a list.
        """
        indexed_solution = {}
        parameter_solution = {}
        for com in self.model.component_objects():
            if com.ctype not in {pyo.Var, pyo.Param, pyo.Objective, pyo.Expression}:
                continue
            if names is not None and com.name not in names:
                continue  # Only include names that where asked for
            if com.is_indexed():
                indexed_solution[com.name] = [pyo.value(v) for v in com.values()]
            else:
                parameter_solution[com.name] = pyo.value(com)
        return indexed_solution, parameter_solution

    @property
    def start_value_mapping(self) -> dict[str, str]:
        """Mapping of initial-condition Param names to their corresponding Expression names.

        Subclasses that should be compatible with :class:`~eta_ctrl.envs.PyomoSimEnv` must define
        a ``_start_value_mapping`` class attribute (e.g. ``_start_value_mapping = {"temp0": "temp_expression"}``).

        :raises AttributeError: If the subclass does not define ``_start_value_mapping``.
        """
        mapping = getattr(self, "_start_value_mapping", None)
        if mapping is None:
            msg = f"Tried to access 'self._start_value_mapping' from '{self.__class__.__name__}', but it doesn't exist."
            raise AttributeError(msg)
        return mapping

    def check_pyomo_sim_compatibility(self, ext_outputs: list[str]) -> None:
        """Validate that this model is compatible with :class:`~eta_ctrl.envs.PyomoSimEnv`.

        Checks that every external output (defined in the state config) has a corresponding entry
        in :attr:`start_value_mapping`, that the mapped Param is a scalar :class:`pyo.Param`,
        and the mapped Expression is an indexed :class:`pyo.Expression`.


        :param ext_outputs: External output names from the environment's StateConfig.
        :raises AttributeError: If ``_start_value_mapping`` is not defined.
        :raises KeyError: If an external output is missing from the expression mapping.
        :raises ValueError: If a mapped component does not exist in the concrete model.
        :raises TypeError: If a component has the wrong Pyomo type (e.g. Var instead of Param).
        """
        for ext_output in ext_outputs:
            if ext_output not in self.start_value_mapping:
                msg = f"Missing '{ext_output}' in start_value_mapping of '{self.__class__.__name__}'"
                raise KeyError(msg)

            com = self.model.component(ext_output)
            if com is None:
                msg = f"Component {ext_output} does not exist in '{self.__class__.__name__}'"
                raise ValueError(msg)

            if not isinstance(com, pyo.Param):
                msg = f"Component {com} must be of type 'Param', but is '{type(com).__name__}'"
                raise TypeError(msg)
            if com.is_indexed():
                msg = f"Component {com} must not be indexed, use 'ScalarParam' instead."
                raise TypeError(msg)

            expr_name = self.start_value_mapping[ext_output]
            com_expr = self.model.component(expr_name)
            if com_expr is None:
                msg = f"Component {expr_name} does not exist in '{self.__class__.__name__}'"
                raise ValueError(msg)

            if not isinstance(com_expr, pyo.Expression):
                msg = f"Component {com_expr} must be of type 'Expression', but is '{type(com_expr).__name__}'"
                raise TypeError(msg)
            if not com_expr.is_indexed():
                msg = f"Component {com_expr} must be indexed to retrieve the second value."
                raise TypeError(msg)
