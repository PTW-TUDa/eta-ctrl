from __future__ import annotations

import abc
import inspect
import pathlib
import time
from datetime import datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from eta_ctrl import timeseries
from eta_ctrl.util import csv_export
from eta_ctrl.util.utils import timestep_to_seconds

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from typing import Any

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.envs.state import StateConfig
    from eta_ctrl.util.type_annotations import ObservationType, Path, StepResult, TimeStep


log = getLogger(__name__)


class BaseEnv(Env, abc.ABC):
    """Abstract environment definition, providing some basic functionality for concrete environments to use.

    The class implements and adapts functions from gymnasium.Env. It provides additional functionality as required by
    the ETA Ctrl framework and should be used as the starting point for new environments.

    The initialization of this superclass performs many of the necessary tasks, required to specify a concrete
    environment. Read the documentation carefully to understand, how new environments can be developed, building on
    this starting point.

    There are some class attributes that must be set and some methods that must be implemented to satisfy the interface.
    This is required to create concrete environments.
    The required class attributes are:

        - **version**: Version number of the environment.
        - **description**: Short description string of the environment.

    The gymnasium interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**
        - **render()**

    .. note::
        Subclasses should implement the private _step and _reset methods rather than
        overriding the public step and reset methods. The public methods handle the
        Gymnasium interface and state management automatically.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback that should be called after each episode.
    :param state_modification_callback: callback that should be called after state setup, before logging the state.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Version of the environment.

        Needs to be implemented for each subclass as a class attribute.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Long description of the environment.

        Needs to be implemented for each subclass as a class attribute.
        """
        raise NotImplementedError

    def __init__(
        self,
        env_id: int,
        config_run: ConfigRun,
        state_config: StateConfig,
        verbose: int = 2,
        callback: Callable | None = None,
        state_modification_callback: Callable | None = None,
        seed: int | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        sim_steps_per_sample: int | str = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        #: Verbosity level used for logging.
        self.verbose: int = verbose
        log.setLevel(int(verbose * 10))

        # Set some standard path settings
        #: Information about the optimization run and information about the paths.
        #: For example, it defines path_results and path_scenarios.
        self.config_run: ConfigRun = config_run

        #: Callback can be used for logging and plotting.
        self.callback: Callable | None = callback
        #: Callback can be used for modifying the state at each time step.
        self.state_modification_callback: Callable | None = state_modification_callback

        #: ID of the environment (useful for vectorized environments).
        self.env_id: int = int(env_id)
        #: Render mode for rendering the environment
        self.render_mode: str | None = render_mode

        #: Duration of one episode in seconds.
        self.episode_duration: float = timestep_to_seconds(episode_duration)
        #: Sampling time (interval between optimization time steps) in seconds.
        self.sampling_time: float = timestep_to_seconds(sampling_time)
        #: Number of time steps (of width sampling_time) in each episode.
        self.n_episode_steps: int = int(self.episode_duration // self.sampling_time)
        #: Duration of the scenario for each episode (for total time imported from csv).
        self.scenario_duration: float = self.episode_duration + self.sampling_time

        #: Beginning time of the scenario.
        self.scenario_time_begin: datetime
        if isinstance(scenario_time_begin, datetime):
            self.scenario_time_begin = scenario_time_begin
        else:
            self.scenario_time_begin = datetime.strptime(scenario_time_begin, "%Y-%m-%d %H:%M")
        #: Ending time of the scenario (should be in the format %Y-%m-%d %H:%M).
        self.scenario_time_end: datetime
        if isinstance(scenario_time_end, datetime):
            self.scenario_time_end = scenario_time_end
        else:
            self.scenario_time_end = datetime.strptime(scenario_time_end, "%Y-%m-%d %H:%M")

        # Check if scenario begin and end times make sense
        if self.scenario_time_begin > self.scenario_time_end:
            msg = "Start time of the scenario should be smaller than or equal to end time."
            raise ValueError(msg)

        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample: int = int(sim_steps_per_sample)

        #: State Configuration for defining State Variables.
        self.state_config: StateConfig = state_config
        self.action_space, self.observation_space = self.state_config.continuous_spaces()

        if seed is not None:
            # Initialize random generator
            Env.reset(self, seed=seed)
            self.action_space.seed(seed=seed)
            self.observation_space.seed(seed=seed)

        self._init_attributes()

    def _init_attributes(self) -> None:
        """Initialize environment attributes that don't depend on constructor arguments."""

        #: Path of the environment file.
        self.path_env: pathlib.Path
        for f in inspect.stack():
            if "__class__" in f.frame.f_locals and f.frame.f_locals["__class__"] is self.__class__:
                self.path_env = pathlib.Path(f.filename).parent

        # Store data logs and log other information
        #: Episode timer (stores the start time of the episode).
        self.episode_timer: float = time.time()
        #: Current state of the environment.
        self.state: dict[str, np.ndarray]
        #: Additional state information to append to the state during stepping and reset
        self.additional_state: dict[str, float] | None = None
        #: Log of the environment state.
        self.state_log: list[dict[str, np.ndarray]] = []
        #: Log of the environment state over multiple episodes.
        self.state_log_longtime: list[list[dict[str, np.ndarray]]] = []

        #: The time series DataFrame contains all time series scenario data. It can be filled by the
        #: import_scenario method.
        self.timeseries: pd.DataFrame = pd.DataFrame()
        #: Data frame containing the currently valid range of time series data.
        self.ts_current: pd.DataFrame = pd.DataFrame()

        #: Number of completed episodes.
        self.n_episodes: int = 0
        #: Current step of the model (number of completed steps) in the current episode.
        self.n_steps: int = 0
        #: Current step of the model (total over all episodes).
        self.n_steps_longtime: int = 0

    @property
    def run_name(self) -> str:
        #: Name of the current optimization run.
        return self.config_run.name

    @property
    def path_results(self) -> pathlib.Path:
        #: Path for storing results.
        return self.config_run.path_results

    @property
    def path_scenarios(self) -> pathlib.Path | None:
        #: Path for the scenario data.
        return self.config_run.path_scenarios

    def import_scenario(self, *scenario_paths: Mapping[str, Any], prefix_renamed: bool = True) -> pd.DataFrame:
        """Load data from csv into self.timeseries_data by using scenario_from_csv.

        :param scenario_paths: One or more scenario configuration dictionaries (or a list of dicts), which each contain
            a path for loading data from a scenario file. The dictionary should have the following structure, with <X>
            denoting the variable value:

            .. note ::
                [{*path*: <X>, *prefix*: <X>, *interpolation_method*: <X>, *resample_method*: <X>,
                *scale_factors*: {col_name: <X>}, *rename_cols*: {col_name: <X>}, *infer_datetime_cols*: <X>,
                *time_conversion_str*: <X>}]

            * **path**: Path to the scenario file (relative to scenario_path).
            * **prefix**: Prefix for all columns in the file, useful if multiple imported files
              have the same column names.
            * **interpolation_method**: A pandas interpolation method, required if the frequency of
              values must be increased in comparison to the files' data. (e.g.: 'linear' or 'pad').
            * **scale_factors**: Scaling factors for specific columns. This can be useful for
              example, if a column contains data in kilowatt and should be imported in watts.
              In this case, the scaling factor for the column would be 1000.
            * **rename_cols**: Mapping of column names from the file to new names for the imported
              data.
            * **infer_datetime_cols**: Number of the column which contains datetime data. If this
              value is not present, the time_conversion_str variable will be used to determine
              the datetime format.
            * **time_conversion_str**: Time conversion string, determining the datetime format
              used in the imported file (default: %Y-%m-%d %H:%M).
        :param prefix_renamed: Determine whether the prefix is also applied to renamed columns.
        :return: Data Frame of the imported and formatted scenario data.
        """
        paths = []
        prefix = []
        int_methods = []
        scale_factors = []
        rename_cols = {}
        infer_datetime_from = []
        time_conversion_str = []

        for path in scenario_paths:
            paths.append(self.path_scenarios / path["path"])
            prefix.append(path.get("prefix", None))
            int_methods.append(path.get("interpolation_method", None))
            scale_factors.append(path.get("scale_factors", None))
            (rename_cols.update(path.get("rename_cols", {})),)
            infer_datetime_from.append(path.get("infer_datetime_cols", "string"))
            time_conversion_str.append(path.get("time_conversion_str", "%Y-%m-%d %H:%M"))

        self.ts_current = timeseries.scenario_from_csv(
            paths=paths,
            resample_time=self.sampling_time,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            total_time=self.scenario_duration,
            random=self.np_random,
            interpolation_method=int_methods,
            scaling_factors=scale_factors,
            rename_cols=rename_cols,
            prefix_renamed=prefix_renamed,
            infer_datetime_from=infer_datetime_from,
            time_conversion_str=time_conversion_str,
        )

        return self.ts_current

    def get_scenario_state(self) -> dict[str, Any]:
        """Get scenario data for the current time step of the environment, as specified in state_config. This assumes
        that scenario data in self.ts_current is available and scaled correctly.

        :return: Scenario data for current time step.
        """
        scenario_state = {}
        for scen in self.state_config.scenarios:
            scenario_state[scen] = self.ts_current[self.state_config.map_scenario_ids[scen]].iloc[self.n_steps]

        return scenario_state

    @abc.abstractmethod
    def _step(self) -> tuple[float, bool, bool, dict]:
        """Abstract method to perform one internal time step.

        This private method must be implemented by subclasses to update the internal
        state dictionary and return step results. It should work with the internal
        state rather than returning observations directly.

        :return: Tuple of (reward, terminated, truncated, info)

        :meta public:
        """

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        This method handles the public interface for the step operation. It validates actions,
        calls the private _step method implemented by subclasses, manages state updates, and
        returns the formatted results.

        It also updates the state log and calls the state modification callback.

        :param action: Actions taken by the agent.
        :return: The return value represents the state of the environment after the step was performed:

            * **observations**: A dictionary with new observation values as defined by the
                observation space, automatically extracted from the internal state.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by EtaCtrl.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
              bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user
              needs to call the `reset` function.
            * **info**: Provide some additional info about the state of the environment. The contents of this may be
              used for logging purposes in the future but typically do not currently serve a purpose.
        """
        self._reset_state()
        self._actions_valid(action)

        self.set_action(action=action)
        # Perform the actual step in the environment
        reward, terminated, truncated, info = self._step()
        self.n_steps += 1
        # Execute optional state modification callback function
        if self.state_modification_callback:
            self.state_modification_callback()

        self.state_log.append(self.state)

        # Render the environment at each step
        if self.render_mode is not None:
            self.render()
        return self.get_observations(), reward, terminated, truncated, info

    def _actions_valid(self, action: np.ndarray | dict) -> None:
        """Check whether the actions are within the specified action space.

        :param action: Actions taken by the agent.
        :raise: RuntimeError, when the actions are not inside of the action space.
        """
        if not self.action_space.contains(action):
            error_msg = self._build_action_error_message(action)
            raise RuntimeError(error_msg)

    def _build_action_error_message(self, action: np.ndarray | dict) -> str:
        """Build a detailed error message explaining why the action is invalid.

        :param action: The invalid action that was provided.
        :return: Detailed error message string.
        """
        error_parts = ["Action validation failed!"]
        error_parts.append(f"\nReceived action: {self._format_array(action)}")
        error_parts.append(f"Action space: {self.action_space}")

        # Delegate to specific validators based on space type
        if isinstance(self.action_space, spaces.Box):
            error_parts.extend(self._validate_box_action(cast("np.ndarray", action), self.action_space))
        elif isinstance(self.action_space, spaces.Discrete):
            error_parts.extend(self._validate_discrete_action(cast("np.ndarray", action), self.action_space))
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            error_parts.extend(self._validate_multi_discrete_action(cast("np.ndarray", action), self.action_space))
        elif isinstance(self.action_space, spaces.Dict):
            error_parts.extend(self._validate_dict_action(action, self.action_space))
        else:
            error_parts.append("\nThe action does not match the expected action space type.")

        return "\n".join(error_parts)

    def _add_shape_error(self, errors: list[str], expected: tuple, received: tuple) -> None:
        """Add shape mismatch error details to error list

        :param errors: List to append error messages to.
        :param expected: Expected shape.
        :param received: Received shape.
        """
        errors.append("\nShape mismatch:")
        errors.append(f"  Expected: {expected}")
        errors.append(f"  Received: {received}")
        if len(expected) == 1 and len(received) == 1:
            errors.append(f"  → Expected {expected[0]} action(s), but received {received[0]} action(s)")

    def _add_violations(self, errors: list[str], violations: list[str], violation_type: str = "Bound") -> None:
        """Add violation details to error list with truncationS

        :param errors: List to append error messages to.
        :param violations: List of violation messages.
        :param violation_type: Type of violation (e.g., "Bound", "Value").
        """
        if violations:
            errors.append(f"\n{violation_type} violations ({len(violations)} found):")
            errors.extend(violations[:10])
            if len(violations) > 10:
                errors.append(f"  ... and {len(violations) - 10} more violation(s)")

    def _validate_box_action(self, action: np.ndarray, space: spaces.Box) -> list[str]:
        """Validate Box space action and return specific error details.

        :param action: The action to validate.
        :param space: The Box space to validate against.
        :return: List of error message parts.
        """
        errors: list[str] = []

        # Check shape
        if action.shape != space.shape:
            self._add_shape_error(errors, space.shape, action.shape)
            return errors

        # Check dtype compatibility (warn but don't fail for float32 vs float64)
        if action.dtype != space.dtype and not (
            np.issubdtype(action.dtype, np.floating) and np.issubdtype(space.dtype, np.floating)
        ):
            errors.append("\nData type mismatch:")
            errors.append(f"  Expected: {space.dtype}")
            errors.append(f"  Received: {action.dtype}")

        # Check bounds
        violations = []
        action_flat = action.flatten()
        low_flat = np.broadcast_to(space.low, space.shape).flatten()
        high_flat = np.broadcast_to(space.high, space.shape).flatten()

        for idx, (val, low, high) in enumerate(zip(action_flat, low_flat, high_flat, strict=False)):
            if val < low:
                violations.append(f"  - action[{idx}] = {val:.6g} is below minimum bound of {low:.6g}")
            elif val > high:
                violations.append(f"  - action[{idx}] = {val:.6g} exceeds maximum bound of {high:.6g}")

        self._add_violations(errors, violations, "Bound")
        return errors

    def _validate_discrete_action(self, action: np.ndarray | int, space: spaces.Discrete) -> list[str]:
        """Validate Discrete space action and return specific error details.

        :param action: The action to validate.
        :param space: The Discrete space to validate against.
        :return: List of error message parts.
        """
        errors: list[str] = []

        # Convert to int if it's an array
        if isinstance(action, np.ndarray):
            if action.size != 1:
                self._add_shape_error(errors, (1,), action.shape)
                return errors
            action_val = int(action.item())
        else:
            action_val = int(action)

        # Check bounds
        if action_val < space.start or action_val >= space.start + space.n:
            errors.append("\nValue out of range:")
            errors.append(f"  Valid range: [{space.start}, {space.start + space.n - 1}]")
            errors.append(f"  Received: {action_val}")

        return errors

    def _validate_multi_discrete_action(self, action: np.ndarray, space: spaces.MultiDiscrete) -> list[str]:
        """Validate MultiDiscrete space action and return specific error details.

        :param action: The action to validate.
        :param space: The MultiDiscrete space to validate against.
        :return: List of error message parts.
        """
        errors: list[str] = []

        # Check shape
        if action.shape != space.nvec.shape:
            self._add_shape_error(errors, space.nvec.shape, action.shape)
            return errors

        # Check individual action values
        violations = []
        action_flat = action.flatten()
        nvec_flat = space.nvec.flatten()
        start_flat = np.broadcast_to(space.start, space.nvec.shape).flatten()

        for idx, (val, n, start) in enumerate(zip(action_flat, nvec_flat, start_flat, strict=False)):
            if val < start or val >= start + n:
                violations.append(f"  - action[{idx}] = {val} is outside valid range [{start}, {start + n - 1}]")

        self._add_violations(errors, violations, "Value")
        return errors

    def _validate_dict_action(self, action: np.ndarray | dict, space: spaces.Dict) -> list[str]:
        """Validate Dict space action and return specific error details.

        :param action: The action to validate.
        :param space: The Dict space to validate against.
        :return: List of error message parts.
        """
        errors: list[str] = []
        errors.append("\nDict action space validation failed.")
        errors.append(f"  Expected a dictionary with keys: {list(space.spaces.keys())}")

        if not isinstance(action, dict):
            errors.append(f"  Received type: {type(action).__name__}")
            errors.append("  → Dict action spaces require actions to be dictionaries, not arrays")
        else:
            missing_keys = set(space.spaces.keys()) - set(action.keys())
            extra_keys = set(action.keys()) - set(space.spaces.keys())

            if missing_keys:
                errors.append(f"  Missing keys: {list(missing_keys)}")
            if extra_keys:
                errors.append(f"  Unexpected keys: {list(extra_keys)}")

        return errors

    def _format_array(self, arr: np.ndarray | dict, max_items: int = 10) -> str:
        """Format a numpy array or dict for display in error messages.

        :param arr: Array or dict to format.
        :param max_items: Maximum number of items to display before truncating.
        :return: Formatted string representation.
        """
        # Handle dict actions (for Dict action spaces)
        if isinstance(arr, dict):
            return str(arr)

        # Handle numpy arrays - cast needed for mypy type narrowing
        arr = cast("np.ndarray", arr)

        if arr.size <= max_items:
            return str(arr)

        # For large arrays, show first few and last few elements
        arr_flat = arr.flatten()
        first_items = arr_flat[: max_items // 2]
        last_items = arr_flat[-(max_items // 2) :]
        formatted = f"[{' '.join(f'{x:.6g}' for x in first_items)} ... {' '.join(f'{x:.6g}' for x in last_items)}]"
        return f"{formatted} (shape: {arr.shape}, dtype: {arr.dtype})"

    def _reset_state(self) -> None:
        """Take some initial values and create a new environment state object, stored in self.state."""
        self.state = {}
        if self.additional_state is not None:
            additional_state = {name: np.array([value]) for name, value in self.additional_state.items()}
            self.state.update(additional_state)

    def _terminated(self) -> bool:
        """Check if the episode is over or not using the number of steps (n_steps) and the total number of
        steps in an episode (n_episode_steps).

        :return: boolean showing, whether the episode is terminated.
        """
        return self.n_steps >= self.n_episode_steps

    @abc.abstractmethod
    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Abstract reset method that must be implemented by subclasses.

        :meta public:
        """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Reset the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter.

        :param seed: The seed for initializing any randomized components of the state.
                     Subclasses should use this for reproducible randomness in their state init
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)

        :return: Tuple of observation and info. The observation of the initial state will be an element of
                :attr:`observation_space` (typically a numpy array) and is analogous to the observation returned by
                :meth:`step`. Info is a dictionary containing auxiliary information complementing ``observation``. It
                should be analogous to the ``info`` returned by :meth:`step`.
        """
        # Reset state_log and counters
        if self.n_steps > 0:
            self._reset_episode()
        # Reset state
        self._reset_state()
        # Set rng seed
        Env.reset(self, seed=seed)
        # Set initial observations in child class
        info = self._reset(options=options)
        # Execute optional state modification callback function
        if self.state_modification_callback:
            self.state_modification_callback()

        self.state_log.append(self.state)

        # Render the environment when calling the reset function
        if self.render_mode is not None:
            self.render()
        return self.get_observations(), info

    def _reduce_state_log(self) -> list[dict[str, np.ndarray]]:
        """Remove unwanted parameters from state_log before storing in state_log_longtime.

        :return: The return value is a list of dictionaries,
         where the parameters that should not be stored were removed
        """
        allowed_keys = set(self.state_config.add_to_state_log)
        return [{k: v for k, v in entry.items() if k in allowed_keys} for entry in self.state_log]

    def _reset_episode(self) -> None:
        """Store episode statistics and reset episode counters."""
        if self.callback is not None:
            self.callback(self)

        # Store some logging data
        self.n_episodes += 1

        # store reduced_state_log in state_log_longtime
        self.state_log_longtime.append(self._reduce_state_log())
        self.n_steps_longtime += self.n_steps

        # Reset episode variables
        self.n_steps = 0
        self.episode_timer = time.time()
        self.state_log = []

    @abc.abstractmethod
    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.
        """
        msg = "Cannot close an abstract Environment."
        raise NotImplementedError(msg)

    @abc.abstractmethod
    def render(self) -> None:
        """Render the environment.

        The set of supported modes varies per environment. Some environments do not support rendering at
        all. By convention in Farama *gymnasium*, if mode is:

            * human: render to the current display or terminal and return nothing. Usually for human consumption.
            * rgb_array: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
              suitable for turning into a video.
            * ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        """
        msg = "Cannot render an abstract Environment."
        raise NotImplementedError(msg)

    @classmethod
    def get_info(cls) -> tuple[str, str]:
        """Get info about environment.

        :return: Tuple of version and description.
        """
        return cls.version, cls.description  # type: ignore[return-value]

    def __str__(self) -> str:
        """Human-readable string representation of BaseEnv."""
        env_class = self.__class__.__name__
        n_actions = len(self.state_config.actions)
        n_observations = len(self.state_config.observations)
        status = f"Episode {self.n_episodes}, Step {self.n_steps}/{self.n_episode_steps}"

        return f"{env_class}(id={self.env_id}, {n_actions} actions, {n_observations} observations, {status})"

    def __repr__(self) -> str:
        """Developer-friendly string representation of BaseEnv."""
        env_class = self.__class__.__name__
        return (
            f"{env_class}(env_id={self.env_id}, run_name='{self.run_name}', "
            f"n_episodes={self.n_episodes}, n_steps={self.n_steps}, "
            f"episode_duration={self.episode_duration}, sampling_time={self.sampling_time})"
        )

    def export_state_log(
        self,
        path: Path,
        names: Sequence[str] | None = None,
        *,
        sep: str = ";",
        decimal: str = ".",
    ) -> None:
        """Extension of csv_export to include timeseries on the data.

        :param names: Field names used when data is a Matrix without column names.
        :param sep: Separator to use between the fields.
        :param decimal: Sign to use for decimal points.
        """
        start_time = datetime.fromtimestamp(self.episode_timer)
        step = self.sampling_time / self.sim_steps_per_sample
        timerange = [start_time + timedelta(seconds=(k * step)) for k in range(len(self.state_log))]
        csv_export(path=path, data=self.state_log, index=timerange, names=names, sep=sep, decimal=decimal)

    def get_observations(self) -> dict[str, np.ndarray]:
        """Gather observations from the state.

        :raises KeyError: Observation is not available in state
        :return: Filtered observations as a dictionary.
        :rtype: dict[str, np.ndarray]
        """
        observations = {}
        for name in self.state_config.observations:
            try:
                observations[name] = self.state[name]
            except KeyError as e:
                msg = f"{e!s} is unavailable in environment state."
                raise KeyError(msg) from e
        return observations

    def get_external_inputs(self) -> dict[str, float]:
        """Gather external inputs from the state.
        Uses scalar values instead of numpy arrays for values.

        :raises KeyError: External input is not available in state
        :raises ValueError: External input value is not scalar
        :return: Filtered external inputs with external id as keys.
        :rtype: dict[str, float]
        """
        external_inputs = {}
        for name in self.state_config.ext_inputs:
            ext_id = self.state_config.map_ext_ids[name]
            state_var = self.state_config.vars[name]
            try:
                scaled_value = self.state[name].item()
            except KeyError as e:
                msg = f"{e!s} is unavailable in environment state."
                raise KeyError(msg) from e
            except ValueError as e:
                msg = "External Inputs can't have multiple values"
                raise ValueError(msg) from e
            external_inputs[ext_id] = scaled_value / state_var.ext_scale_mult - state_var.ext_scale_add
        return external_inputs

    def set_action(self, action: np.ndarray | dict[str, np.ndarray]) -> None:
        """Set action values in the state.

        :param action: Actions to be set.
        :type action: np.ndarray | dict[str, np.ndarray]
        """
        iterator: Iterator
        if isinstance(action, np.ndarray):
            iterator = zip(self.state_config.actions, action, strict=True)
        else:
            iterator = iter(action.items())

        for name, value in iterator:
            val = value if isinstance(value, np.ndarray) else np.array([value])
            self.state[name] = val

    def set_external_outputs(self, external_outputs: dict[str, float]) -> None:
        """Set external outputs in the state.
        Accepts scalars instead of numpy arrays as values.

        :param external_outputs: Dict of external outputs with external_ids as keys.
        :type external_outputs: dict[str, float]
        :raises KeyError: Received an unknown external id
        """
        for ext_id, unscaled_value in external_outputs.items():
            name = self.state_config.rev_ext_ids.get(ext_id)

            # Check if name exists and if it is an output
            if name is None or name not in self.state_config.ext_outputs:
                if name is None:
                    msg = "Received unknown name for external outputs"
                else:
                    msg = f"{name} is not configured as an external output."
                raise KeyError(msg)
            state_var = self.state_config.vars[name]
            scaled_value = (unscaled_value + state_var.ext_scale_add) * state_var.ext_scale_mult

            self.state[name] = np.array([scaled_value])
