from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING

from eta_nexus.connections import LiveConnect

from eta_ctrl.envs import BaseEnv

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from typing import Any

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.util.type_annotations import Path, TimeStep

log = getLogger(__name__)


class LiveEnv(BaseEnv, abc.ABC):
    """Base class for Live environments.

    The class will create an ETA Nexus ConnectionManager instance
    and provide facilities to automatically read step results and reset the connection.

    Additionally to required class attribute from `BaseEnv`, `LiveEnv` requires
    the name of the connection manager configuration file as a class attribute:

      - **config_name**: Name of the connection manager configuration.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param max_errors: Maximum number of connection errors before interrupting the optimization process.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def config_name(self) -> str:
        """Name of the connection manager configuration.

        Needs to be implemented for each subclass as a class attribute.
        """
        raise NotImplementedError

    def __init__(
        self,
        env_id: int,
        config_run: ConfigRun,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        max_errors: int = 10,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
            **kwargs,
        )
        #: Instance of the Live Connector.
        self.connection_manager: LiveConnect
        #: Path or Dict to initialize the live connector.
        self.connection_manager_config: Path | Sequence[Path] | dict[str, Any] | None = (
            self.path_env / f"{self.config_name}.json"
        )
        #: Maximum error count when connections in live connector are aborted.
        self.max_error_count: int = max_errors

    def __str__(self) -> str:
        """Human-readable string representation of LiveEnv."""
        base_str = super().__str__()
        config_name = self.config_name
        return f"{base_str}, Live config: {config_name}"

    def __repr__(self) -> str:
        """Developer-friendly string representation of LiveEnv."""
        base_repr = super().__repr__()
        # Remove the closing parenthesis to add our info
        base_repr = base_repr.rstrip(")")
        return f"{base_repr}, config_name='{self.config_name}', max_error_count={self.max_error_count})"

    def _init_connection_manager(self, files: Path | Sequence[Path] | dict[str, Any] | None = None) -> None:
        """Initialize the live connector object. Make sure to call _names_from_state before this or to otherwise
        initialize the names array.

        :param files: Path or Dict to initialize the connection directly from JSON configuration files or a config
            dictionary.
        """
        _files = self.connection_manager_config if files is None else files
        self.connection_manager_config = _files

        if _files is None:
            msg = "Configuration files or a dictionary must be specified before the connector can be initialized."
            raise TypeError(msg)

        if isinstance(_files, dict):
            self.connection_manager = LiveConnect.from_dict(
                step_size=self.sampling_time,
                max_error_count=self.max_error_count,
                **_files,
            )
        else:
            self.connection_manager = LiveConnect.from_config(
                files=_files, step_size=self.sampling_time, max_error_count=self.max_error_count
            )

    def _step(self) -> tuple[float, bool, bool, dict]:
        """Perform one internal time step and return core step results.

        This is called for every event or for every time step during the simulation/optimization run.
        It should utilize the actions as supplied by the agent to determine the new
        state of the environment, which are available in the state dictionary.

        This also updates self.state and self.state_log to store current state information.

        .. note::
            This function always returns 0 reward. Therefore, it must be extended if it is to be used with reinforcement
            learning agents. If you need to manipulate actions (discretization, policy shaping, ...) do this before
            calling this function. If you need to manipulate observations and rewards, do this after calling this
            function.

        :return: A tuple containing:

            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
                the reset function will automatically be called by the agent or by EtaCtrl.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
                bounds. Can be used to end the episode prematurely before a terminal state is reached.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
                be used for logging purposes in the future but typically do not currently serve a purpose.

        .. note::
            Stable Baselines3 combines terminated and truncated with a logical OR to trigger
            the automatic environment reset. Implement both flags for compatibility.

        :meta public:
        """
        # Set the external inputs in the live connector and read out the external outputs
        results = self.connection_manager.step(value=self.get_external_inputs())

        # Update scenario data, do one time step in the live connector and store the results.
        self.state.update(
            self.get_scenario_state()
        )  # TODO: change in MR https://git.ptw.maschinenbau.tu-darmstadt.de/eta-fabrik/public/eta-ctrl/-/merge_requests/22

        self.set_external_outputs(external_outputs=results)

        return 0, self._terminated(), False, {}

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset. When using the environment in conjunction with
        *stable_baselines3*, the vectorized environment will take care of seeding your custom environment automatically.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        :param seed: The seed for initializing any randomized components of the state.
                     Subclasses should use this for reproducible randomness in their state init
        :param options: Additional information to specify how the environment is reset (optional,
                        depending on the specific environment) (default: None)

        :return: Info dictionary containing information about the initial state.
                 The initial observations are automatically filtered from the internal state
                 by the public reset method and must not be returned here.

        .. note::
            The base implementation initializes external outputs from the live connector
            without using the seed. Subclasses should use the seed parameter
            for any additional randomized state observations they implement.

        :meta public:
        """
        self._init_connection_manager()

        # Read out the start conditions from LiveConnect and store the results
        start_obs_names = [self.state_config.map_ext_ids[name] for name in self.state_config.ext_outputs]
        results = self.connection_manager.read(*start_obs_names)

        self.set_external_outputs(external_outputs=results)

        # Update scenario data
        self.state.update(self.get_scenario_state())

        return {}

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the connection_manager environment is to do nothing.
        """
        if hasattr(self, "connection_manager"):
            self.connection_manager.close()
