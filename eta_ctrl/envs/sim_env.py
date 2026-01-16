from __future__ import annotations

import abc
import time
from logging import getLogger
from typing import TYPE_CHECKING

from eta_ctrl.envs import BaseEnv
from eta_ctrl.simulators import FMUSimulator

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Mapping
    from typing import Any

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.util.type_annotations import TimeStep

log = getLogger(__name__)


class SimEnv(BaseEnv, abc.ABC):
    """Base class for FMU Simulation models environments.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param sim_steps_per_sample: Number of simulation steps to perform during every sample.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def fmu_name(self) -> str:
        """Name of the FMU file."""
        return ""

    def __init__(
        self,
        env_id: int,
        config_run: ConfigRun,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        model_parameters: Mapping[str, Any] | None = None,
        sim_steps_per_sample: int | str = 1,
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

        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample: int = int(sim_steps_per_sample)

        #: The FMU is expected to be placed in the same folder as the environment
        self.path_fmu: pathlib.Path = self.path_env / (self.fmu_name + ".fmu")

        #: Configuration for the FMU model parameters, that need to be set for initialization of the Model.
        self.model_parameters: Mapping[str, int | float] | None = model_parameters

        #: Instance of the FMU. This can be used to directly access the eta_ctrl.FMUSimulator interface.
        self.simulator: FMUSimulator

    def __str__(self) -> str:
        """Human-readable string representation of SimEnv."""
        base_str = super().__str__()
        fmu_name = self.fmu_name
        return f"{base_str}, FMU: {fmu_name}"

    def __repr__(self) -> str:
        """Developer-friendly string representation of SimEnv."""
        base_repr = super().__repr__()
        # Remove the closing parenthesis to add our info
        base_repr = base_repr.rstrip(")")
        return f"{base_repr}, fmu_name='{self.fmu_name}', sim_steps_per_sample={self.sim_steps_per_sample})"

    def _init_simulator(self, init_values: Mapping[str, int | float] | None = None) -> None:
        """Initialize the simulator object. Make sure to call _names_from_state before this or to otherwise initialize
        the names array.

        This can also be used to reset the simulator after an episode is completed. It will reuse the same simulator
        object and reset it to the given initial values.

        :param init_values: Dictionary of initial values for some FMU variables.
        """
        _init_vals = {} if init_values is None else init_values

        if hasattr(self, "simulator") and isinstance(self.simulator, FMUSimulator):
            self.simulator.reset(_init_vals)
        else:
            # Instance of the FMU. This can be used to directly access the eta_ctrl.FMUSimulator interface.
            self.simulator = FMUSimulator(
                self.env_id,
                self.path_fmu,
                start_time=0.0,
                stop_time=self.episode_duration,
                step_size=float(self.sampling_time / self.sim_steps_per_sample),
                names_inputs=[self.state_config.map_ext_ids[name] for name in self.state_config.ext_inputs],
                names_outputs=[self.state_config.map_ext_ids[name] for name in self.state_config.ext_outputs],
                init_values=_init_vals,
            )

    def simulate(self) -> tuple[bool, float]:
        """Perform a simulator step.

        Updates the state with new external outputs from the simulation results.

        :return: Boolean showing whether all simulation steps were successful and time elapsed
                 during simulation.
        """
        # generate FMU input from current state
        step_inputs: dict[str, float] = self.get_external_inputs()

        sim_time_start = time.time()
        step_success = True
        try:
            # We provide output and input names to the FMU so output will be a dictionary
            step_output: dict[str, float] = self.simulator.step(input_values=step_inputs)
        except Exception:
            step_success = False
            log.exception("Simulation failed")

        # stop timer for simulation step time debugging
        sim_time_elapsed = time.time() - sim_time_start

        # save step_outputs into state
        if step_success:
            self.set_external_outputs(step_output)

        return step_success, sim_time_elapsed

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
        step_success, sim_time_elapsed = self._update_state()
        info: dict[str, Any] = {"sim_time_elapsed": sim_time_elapsed}
        # ensure mutual exclusivity of terminated and truncated
        return 0, not step_success, self._truncated() and step_success, info

    def _update_state(self) -> tuple[bool, float]:
        """Take additional_state, execute simulation and get state information from scenario. This function
        updates self.state and increments the step counter.

        .. warning::
            You have to update self.state_log with the entire state before leaving the step
            to store the state information.

        :return: Success of the simulation, time taken for simulation.
        """
        step_success, sim_time_elapsed = False, 0.0
        # simulate one time step and store the results.
        for i in range(self.sim_steps_per_sample):  # do multiple FMU steps in one environment-step
            step_success, sim_time_elapsed = self.simulate()  # only ext inputs are needed

            # Append intermediate simulation results to the state_log
            if i < self.sim_steps_per_sample - 1:
                self.state_log.append(self.state)

        return step_success, sim_time_elapsed

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
            The base implementation initializes external outputs from the FMU without using the seed.
            Subclasses should use the seed parameter for any additional
            randomized state observations they implement.

        :meta public:
        """
        # reset the FMU after every episode with new parameters
        self._init_simulator(self.model_parameters)

        # Read values from the fmu without time step and store the results
        start_obs = [str(self.state_config.map_ext_ids[name]) for name in self.state_config.ext_outputs]
        # We provide output and input names to the FMU so output will be a dictionary
        output: dict[str, float] = self.simulator.read_values(start_obs)
        self.set_external_outputs(output)

        return {}

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the Simulation environment is to close the FMU object.
        """
        self.simulator.close()  # close the FMU
