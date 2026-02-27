from __future__ import annotations

import abc
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import timedelta
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyomo import environ as pyo
from pyomo.core import base as pyo_base

from eta_ctrl.common.export_pyomo import export_pyomo_state
from eta_ctrl.envs import BaseEnv
from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable
    from typing import Any

    from pyomo.opt import SolverResults

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.util.type_annotations import PyoParams, TimeStep


log = getLogger(__name__)


class PyomoEnv(BaseEnv, abc.ABC):
    """Base class for mathematical MPC models. This class can be used in conjunction with the MathSolver agent.
    You need to implement the *_model* method in a subclass and return a *pyomo.AbstractModel* from it.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param prediction_horizon: Duration of the prediction (usually a subsample of the episode duration).
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    def __init__(
        self,
        env_id: int,
        config_run: ConfigRun,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        model_parameters: Mapping[str, Any],
        prediction_horizon: TimeStep | str | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
            **kwargs,
        )

        # Check configuration for MILP compatibility
        # Total duration of one prediction/optimization run when used with the MPC agent.
        # Necessary parameter for the PyomoEnv. If not supplied, raises an error.
        self.prediction_horizon: float
        if prediction_horizon is None:
            msg = "Parameter prediction_horizon is not present in config, but is required for the `PyomoEnv`"
            raise ValueError(msg)

        self.prediction_horizon = float(
            prediction_horizon if not isinstance(prediction_horizon, timedelta) else prediction_horizon.total_seconds()
        )

        if self.prediction_horizon % self.sampling_time != 0:
            msg = (
                "The sampling_time must be a divisor of the prediction_horizon"
                "(prediction_horizon % sampling_time must equal 0)."
            )
            raise ValueError(msg)

        # Make some more settings easily accessible
        #: Number of steps in the prediction (prediction_horizon/sampling_time).
        self.n_prediction_steps: int = int(self.prediction_horizon // self.sampling_time)

        #: Configuration for the MILP model parameters.
        self.model_parameters = model_parameters

        # Set additional attributes with model specific information.
        self._concrete_model: pyo.ConcreteModel | None = None  #: Concrete pyomo model as initialized by _model.

        #: Name of the "time" variable/set in the model (i.e. "T"). This is if the pyomo sets must be re-indexed when
        #:   updating the model between time steps. If this is None, it is assumed that no reindexing of the timeseries
        #:   data is required during updates - this is the default.
        self.time_var: str | None = None

        #: Updating indexed model parameters can be achieved either by updating only the first value of the actual
        #:   parameter itself or by having a separate handover parameter that is used for specifying only the first
        #:   value. The separate handover parameter can be denoted with an appended string. For example, if the actual
        #:   parameter is x.ON then the handover parameter could be x.ON_first. To use x.ON_first for updates, set the
        #:   nonindex_update_append_string to "_first". If the attribute is set to None, the first value of the
        #:   actual parameter (x.ON) would be updated instead.
        self.nonindex_update_append_string: str | None = None

        #: Some models may not use the actual time increment (sampling_time). Instead, they would translate into model
        #:   time increments (each sampling time increment equals a single model time step). This means that indices
        #:   of the model components simply count 1,2,3,... instead of 0, sampling_time, 2*sampling_time, ...
        #:   Set this to true, if model time increments (1, 2, 3, ...) are used. Otherwise, sampling_time will be used
        #:   as the time increment. Note: This is only relevant for the first model time increment, later increments
        #:   may differ.
        self.use_model_time_increments: bool = False

        if not isinstance(self.scenario_manager, CsvScenarioManager):
            msg = "The PyomoEnv needs a CsvScenarioManager, please provide the experiment with scenario files"  # type: ignore[unreachable]
            raise TypeError(msg)
        self.scenario_manager: CsvScenarioManager

    def __str__(self) -> str:
        """Human-readable string representation of PyomoEnv."""
        base_str = super().__str__()
        pred_steps = self.n_prediction_steps
        return f"{base_str}, Prediction steps: {pred_steps}"

    def __repr__(self) -> str:
        """Developer-friendly string representation of PyomoEnv."""
        base_repr = super().__repr__()
        # Remove the closing parenthesis to add our info
        base_repr = base_repr.rstrip(")")
        return (
            f"{base_repr}, prediction_horizon={self.prediction_horizon}, n_prediction_steps={self.n_prediction_steps})"
        )

    @property
    def model(self) -> tuple[pyo.ConcreteModel, list]:
        """The model property is a tuple of the concrete model and the order of the action space. This is used
        such that the MPC algorithm can re-sort the action output. This sorting cannot be conveyed differently through
        pyomo.

        :return: Tuple of the concrete model and the order of the action space.
        """
        if self._concrete_model is None:
            self._concrete_model = self._model()

        return self._concrete_model, self.state_config.actions

    @model.setter
    def model(self, value: pyo.ConcreteModel) -> None:
        """The model attribute setter should be used for returning the solved model.

        :param value: The pyomo.ConcreteModel object which should be used as the model.
        """
        if not isinstance(value, pyo.ConcreteModel):
            msg = "The model attribute can only be set with a pyomo concrete model."
            raise TypeError(msg)
        self._concrete_model = value

    @abc.abstractmethod
    def _model(self) -> pyo.AbstractModel:
        """Create the abstract pyomo model. This is where the pyomo model description should be placed.

        :return: Abstract pyomo model.

        :meta public:
        """
        msg = "The abstract MPC environment does not implement a model."
        raise NotImplementedError(msg)

    def _step(self) -> tuple[float, bool, bool, dict]:
        """Perform one internal time step and return core step results.

        This private method implements the actual environment transition logic. It works
        with the internal self.state dictionary that already includes actions
        and returns the core step results without observations (which are handled by the
        public step method).

        .. note::
            This function always returns 0 reward. Therefore, it must be extended if it is
            to be used with reinforcement learning agents.
            If you need to work with modified actions (e.g., discretized or shaped actions),
            ensure they are processed before reaching this method or handle them within this method
            using the values in self.state.
            If you need to manipulate observations afterwarads, you can do this using the state modification callback.

        :return: A tuple containing:

            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated (bool)**: Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the Vectorizer will call :meth:`reset`.
            * **truncated (bool)**: Whether the truncation condition outside the scope of the MDP is satisfied
                (i.e. the episode ended). Typically, this is a timelimit, but could also be used to indicate an agent
                physically going out of bounds. Can be used to end the episode prematurely before a terminal state is
                reached. If true, the Vectorizer will call :meth:`reset`.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.

        .. note::
            Stable Baselines3 combines terminated and truncated with a logical OR to trigger
            the automatic environment reset. Implement both flags for compatibility.

        :meta public:
        """
        self._reset_state()
        # Update and log current state
        self.update()
        self.state_log.append(self.state)
        reward = pyo.value(next(self.model[0].component_objects(pyo.Objective)))
        return reward, False, self._truncated(), {}

    def update(self, observations: Sequence[Sequence[float | int]] | None = None) -> None:
        """Update the optimization model with observations from another environment.
        New observations are stored in self.state.

        :param observations: Observations from another environment.
        """
        # The timeseries data must be updated for the next time step. The index depends on whether time itself is being
        # shifted. If time is being shifted, the respective variable should be set as "time_var".
        step = int(1 if self.use_model_time_increments else self.sampling_time)
        duration = int(
            self.prediction_horizon // self.sampling_time + 1
            if self.use_model_time_increments
            else self.prediction_horizon
        )
        ts = self.scenario_manager.scenarios.iloc[self.n_steps : self.n_steps + self.n_prediction_steps + 1]
        if self.time_var is not None:
            index = range(self.n_steps * step, duration + (self.n_steps * step), step)
            ts_current = self.pyo_convert_timeseries(
                ts=ts,
                index=tuple(index),
                _add_wrapping_none=False,
            )
            ts_current[self.time_var] = list(index)
            log.debug(
                f"Updated time_var ({self.time_var}) with the set from {index[0]} to "
                f"{index[1]} and steps (sampling time) {self.sampling_time}."
            )
        else:
            index = range(0, duration, step)
            ts_current = self.pyo_convert_timeseries(
                ts=ts,
                index=tuple(index),
                _add_wrapping_none=False,
            )
        updated_params = ts_current

        # Log current optimization window
        log.info(f"Optimization at time step {self.n_steps} of {self.n_episode_steps}.")
        if self.n_steps < self.n_episode_steps:
            time_index = self.scenario_manager.scenarios.index
            time_start = time_index[self.n_steps]
            time_end = time_index[self.n_steps + self.n_prediction_steps + 1]
            log.info(f"Optimization Horizon: {time_start} to {time_end}.")
        else:
            log.info("Last time step reached.")

        self._reset_state()
        for var_name in self.state_config.observations:
            statevar = self.state_config.vars[var_name]
            if not isinstance(statevar.interact_id, int):
                msg = "The interact_id value for observations must be an integer."
                raise TypeError(msg)
            value = None

            # Read values from external environment (for example simulation)
            if observations is not None and statevar.from_interact is True:
                value = round(
                    (observations[0][statevar.interact_id] + statevar.interact_scale_add)
                    * statevar.interact_scale_mult,
                    5,
                )
                self.state[var_name] = np.array([value])
            else:
                # Read additional values from the mathematical model
                for component in self.model[0].component_objects():
                    if component.name == var_name:
                        # Get value for the component from specified index
                        value = round(pyo.value(component[list(component.keys())[int(statevar.index)]]), 5)
                        new_value = value if value is not None else np.nan
                        self.state[var_name] = np.array([new_value])
                        break
                else:
                    log.error(f"Specified observation value {var_name} could not be found.")
            updated_params[var_name] = value
            new_value = value if value is not None else float("nan")
            self.state[var_name] = np.array([new_value])

            log.debug(f"Observed value {var_name}: {value}")

        self.state_log.append(self.state)
        self.pyo_update_params(updated_params, self.nonindex_update_append_string)

    def handle_failed_solve(self, model: pyo.ConcreteModel, result: SolverResults) -> None:
        """This method will try to render the result in case the model could not be solved. It should automatically
        be called by the agent.

        :param model: Current model.
        :param result: Result of the last solution attempt.
        """
        self.model = model
        try:
            self.render()
        except Exception:
            log.exception("Rendering partial results failed")
        self.reset()

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the internal state of the environment and return info dictionary.

        This private method initializes the internal self.state dictionary by reading initial
        values directly from the FMU/simulator. It does not use the seed parameter since the
        initial state is determined by the simulator configuration.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        The public reset method handles the Gymnasium interface including observation filtering
        and proper seeding mechanism.

        :param seed: The seed for initializing any randomized components of the state.
                     Subclasses should use this for reproducible randomness in their state init
        :param options: Additional information to specify how the environment is reset
                (optional, depending on the specific environment) (default: None)

        :return: Info dictionary containing information about the initial state.
                The initial observations are automatically filtered from the internal state
                by the public reset method and must not be returned here.

        .. note::
            The base implementation initializes observations from the pyomo model without using the seed.
            Subclasses should use the seed parameter for any additional
            randomized state observations they implement.

        :meta public:
        """
        if self.n_steps > 0:
            self.model = self._model()

        # Initialize state with the initial observation
        for var_name in self.state_config.observations:
            # Try getting the first value from initialized variables. Use the configured low_value from state_config
            # for all others.
            obs_val = self.pyo_get_component_value(self.model[0].component(var_name), allow_stale=True)
            obs_val = obs_val if obs_val is not None else 0
            self.state[var_name] = np.array([obs_val])

        return {}

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the MPC environment is to do nothing.
        """

    def pyo_component_params(
        self,
        component_name: None | str,
        ts: pd.DataFrame | pd.Series | dict[str, dict] | Sequence | None = None,
        index: pd.Index | Sequence | pyo.Set | None = None,
    ) -> PyoParams:
        """Retrieve parameters for the named component and convert the parameters into the pyomo dict-format.
        If required, timeseries can be added to the parameters and timeseries may be reindexed. The
        pyo_convert_timeseries function is used for timeseries handling. See also *pyo_convert_timeseries*

        :param component_name: Name of the component.
        :param ts: Timeseries for the component.
        :param index: New index for timeseries data. If this is supplied, all timeseries will be copied and
                      reindexed.
        :return: Pyomo parameter dictionary.
        """
        if component_name is None:
            params = self.model_parameters
        elif component_name in self.model_parameters:
            params = self.model_parameters[component_name]
        else:
            params = {}
            log.warning(f"No parameters specified for requested component {component_name}")
        out: PyoParams
        out = {
            param: {None: float(value) if isinstance(value, str) and value in {"inf", "-inf"} else value}
            for param, value in params.items()
        }

        # If component name was specified only look for relevant time series
        if ts is not None:
            out.update(self.pyo_convert_timeseries(ts, index, component_name, _add_wrapping_none=False))

        return {None: out}

    @staticmethod
    def pyo_convert_timeseries(
        ts: pd.DataFrame | pd.Series | dict[str | None, dict[str, Any] | Any] | Sequence,
        index: pd.Index | Sequence | pyo.Set | None = None,
        component_name: str | None = None,
        *,
        _add_wrapping_none: bool = True,
    ) -> PyoParams:
        """Convert a time series data into a pyomo format. Data will be reindexed if a new index is provided.

        :param ts: Timeseries to convert.
        :param index: New index for timeseries data. If this is supplied, all timeseries will be copied and
                      reindexed.
        :param component_name: Name of a specific component that the timeseries is used for. This limits which
                               timeseries are returned.
        :param _add_wrapping_none: Add a "None" indexed dictionary as the top level.
        :return: Pyomo parameter dictionary.
        """
        output: PyoParams = {}
        if index is not None and not isinstance(index, list):
            index = list(index)

        # If part of the timeseries was converted before, make sure that everything is on the same level again.
        _ts: pd.DataFrame | pd.Series | dict[str, Any] | Sequence = (
            ts[None] if isinstance(ts, dict) and None in ts and isinstance(ts[None], Mapping) else ts
        )

        def convert_index(cts: pd.Series | Sequence | Mapping, _index: Sequence[int] | None) -> dict[int, Any]:
            """Take the timeseries and change the index to correspond to _index.

            :param cts: Original timeseries object (with or without index does not matter).
            :param _index: New index.
            :return: New timeseries dictionary with the converted index.
            """
            values = None
            if isinstance(cts, pd.Series):
                values = cts.to_numpy()
            elif isinstance(cts, Sequence):
                values = cts
            elif isinstance(cts, Mapping):
                values = cts.values()

            if _index is not None and values is not None:
                cts = dict(zip(_index, values, strict=False))
            elif _index is not None and values is None:
                msg = "Unsupported timeseries type for index conversion."
                raise ValueError(msg)

            return cts

        if isinstance(_ts, pd.DataFrame | Mapping):
            for key, t in _ts.items():
                # Determine whether the timeseries should be returned, based on the timeseries name and the requested
                #  component name.
                if component_name is not None and "." in key and component_name not in key.split("."):
                    continue
                split_key = key.split(".")[-1]
                # Simple values do not need their index converted...
                if not hasattr(t, "__len__") and np.isreal(t):
                    output[split_key] = {None: t}
                else:
                    output[split_key] = convert_index(t, index)

        elif isinstance(_ts, pd.Series):
            # Determine whether the timeseries should be returned, based on the timeseries name and the requested
            #  component name.
            if (
                component_name is not None
                and isinstance(_ts.name, str)
                and "." in _ts.name
                and component_name in _ts.name.split(".")
            ):
                output[_ts.name.split(".")[-1]] = convert_index(_ts, index)
            elif component_name is None or "." not in _ts.name:
                output[_ts.name] = convert_index(_ts, index)

        else:
            output[None] = convert_index(_ts, index)

        return {None: output} if _add_wrapping_none else output

    def pyo_update_params(
        self,
        updated_params: MutableMapping[str | None, Any],
        nonindex_param_append_string: str | None = None,
    ) -> None:
        """Update model parameters and indexed parameters of a pyomo instance with values given in a dictionary.
        It assumes that the dictionary supplied in updated_params has the correct pyomo format.

        :param updated_params: Dictionary with the updated values.
        :param nonindex_param_append_string: String to be appended to values which are not indexed. This can
            be used if indexed parameters need to be set with values that do not have an index.
        :return: Updated model instance.
        """
        # append string to non indexed values that are used to set indexed parameters.
        if nonindex_param_append_string is not None:
            original_indices = set(updated_params.keys()).copy()
            for param in original_indices:
                component = self.model[0].component(param)
                if (
                    component is not None
                    and (component.is_indexed() or isinstance(component, pyo.Set | pyo.RangeSet))
                    and not isinstance(updated_params[param], Mapping)
                ):
                    updated_params[str(param) + nonindex_param_append_string] = updated_params[param]
                    del updated_params[param]

        for parameter in self.model[0].component_objects():
            parameter_name = str(parameter)
            if parameter_name not in updated_params:
                # last entry is the parameter name for abstract models which are instanced
                parameter_name = parameter_name.split(".")[-1]

            if parameter_name in updated_params:
                if isinstance(parameter, pyo_base.param.ScalarParam | pyo_base.var.ScalarVar):
                    # update all simple parameters (single values)
                    parameter.value = updated_params[parameter_name]
                elif isinstance(parameter, pyo_base.indexed_component.IndexedComponent):
                    # update all indexed parameters (time series)
                    if not isinstance(updated_params[parameter_name], Mapping):
                        parameter[next(parameter)] = updated_params[parameter_name]
                    else:
                        for param_val in list(parameter):
                            parameter[param_val] = updated_params[parameter_name][param_val]

        log.info("Pyomo model parameters updated.")

    def pyo_get_solution(self, names: set[str] | None = None) -> dict[str, float | int | dict[int, float | int]]:
        """Convert the pyomo solution into a more usable format for plotting.

        :param names: Names of the model parameters that are returned.
        :return: Dictionary of {parameter name: value} pairs. Value may be a dictionary of {time: value} pairs which
                 contains one value for each optimization time step.
        """
        solution = {}

        for com in self.model[0].component_objects():
            if com.ctype not in {pyo.Var, pyo.Param, pyo.Objective}:
                continue
            if names is not None and com.name not in names:
                continue  # Only include names that where asked for

            # For simple variables we need just the values, for everything else we want time indexed dictionaries
            if isinstance(com, pyo.ScalarVar | pyo_base.objective.SimpleObjective | pyo_base.param.ScalarParam):
                solution[com.name] = pyo.value(com)
            else:
                solution[com.name] = {}
                if self.use_model_time_increments:
                    for ind, val in com.items():
                        solution[com.name][
                            self.scenario_manager.scenarios.index[self.n_steps].to_pydatetime()
                            + timedelta(seconds=ind * self.sampling_time)
                        ] = pyo.value(val)
                else:
                    for ind, val in com.items():
                        solution[com.name][
                            self.scenario_manager.scenarios.index[self.n_steps].to_pydatetime() + timedelta(seconds=ind)
                        ] = pyo.value(val)

        return solution

    def pyo_get_component_value(
        self, component: pyo.Component, *, at: int = 1, allow_stale: bool = False
    ) -> float | int | None:
        if allow_stale and (
            (getattr(component, "stale", None)) or (getattr(component, "value", None) is component.NoValue)
        ):
            return self.state_config.vars[component.name].low_value

        if isinstance(component, pyo.Var | pyo.RangeSet):
            val = round(pyo.value(component.at(at)), 5)
        elif component.is_indexed() and (
            not hasattr(component, "stale") or (hasattr(component, "stale") and not component.stale)
        ):
            val = round(pyo.value(component[component.index_set().at(at)]), 5)
        else:
            val = round(pyo.value(component), 5)

        return val

    @classmethod
    def create_state(
        cls, model: pyo.ConcreteModel, model_name: str, output_dir: pathlib.Path | str | None = None
    ) -> None:
        """Create both state config and parameters files from a Pyomo model.

        This method creates both a state configuration TOML file (containing variables/observations)
        and a parameters TOML file from a Pyomo ConcreteModel, providing a complete setup for
        Pyomo-based environments.

        :param model: Pyomo ConcreteModel instance.
        :param model_name: Name of the model for identification.
        :param output_dir: Directory where files should be created. If None, uses current working directory.
        """
        # Delegate to the dedicated export function
        export_pyomo_state(model, model_name, output_dir)
