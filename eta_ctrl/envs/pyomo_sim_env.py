from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING

from eta_ctrl.common.export_pyomo import export_pyomo_state
from eta_ctrl.envs import BaseEnv
from eta_ctrl.simulators import PyomoModel
from eta_ctrl.util.utils import import_class_from_module

if TYPE_CHECKING:
    import pathlib
    from typing import Any

    from pyomo import environ as pyo


log = getLogger(__name__)


class PyomoSimEnv(BaseEnv, abc.ABC):
    """Gymnasium environment that simulates state transitions using a Pyomo model without a solver.

    Instead of optimizing over a full prediction horizon, ``PyomoSimEnv`` instantiates the model
    with ``prediction_horizon = sampling_time`` (i.e. one time step) and evaluates Pyomo
    :class:`~pyomo.core.base.expression.Expression` components to compute the next state.

    The model must define an :attr:`~eta_ctrl.simulators.PyomoModel.start_value_mapping` that maps
    initial-condition Param names to their corresponding Expression names. Each step, the environment:

    1. Fixes agent actions in the model at t=0.
    2. Evaluates the mapped Expressions at t=1 to obtain the next state.
    3. Updates the initial-condition Params via :meth:`~eta_ctrl.simulators.PyomoModel.pyo_update_params`
       for the following step.

    This allows reusing the same Pyomo model definition for both MPC optimization (with
    :class:`~eta_ctrl.agents.MpcAgent`) and step-by-step simulation.

    :param args: Positional arguments forwarded to :class:`~eta_ctrl.envs.BaseEnv`.
    :param kwargs: Keyword arguments forwarded to :class:`~eta_ctrl.envs.BaseEnv`.
        May include ``model_parameters`` (dict) which is extracted and passed to the model constructor.
    """

    @property
    @abc.abstractmethod
    def model_import(self) -> str:
        """Dotted import path to the :class:`~eta_ctrl.simulators.PyomoModel` subclass."""
        return ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        target_class: type[PyomoModel] = import_class_from_module(self.model_import, base_class=PyomoModel)

        # Use sampling time as prediction scope, resulting in one time step
        self.model = target_class(
            model_parameters=kwargs.pop("model_parameters", None),
            sampling_time=self.sampling_time,
            prediction_horizon=self.sampling_time,
        )

        try:
            self.model.check_pyomo_sim_compatibility(
                [self.state_config.map_ext_ids[id_] for id_ in self.state_config.ext_outputs]
            )
        except (ValueError, KeyError, TypeError) as e:
            msg = f"PyomoModel '{target_class.__name__}' is not compatible with the PyomoSimEnv, see the documentation."
            raise NotImplementedError(msg) from e

    def __str__(self) -> str:
        """Human-readable string representation of PyomoSimEnv."""
        base_str = super().__str__()
        pyomo_model_type = self.model_import.split(".")[-1]
        return f"{base_str}, PyomoModel: {pyomo_model_type}"

    def __repr__(self) -> str:
        """Developer-friendly string representation of PyomoSimEnv."""
        base_repr = super().__repr__()
        # Remove the closing parenthesis to add our info
        base_repr = base_repr.rstrip(")")
        return f"{base_repr}, model_import='{self.model_import}')"

    def _step(self) -> tuple[float, bool, bool, dict]:
        """Perform one internal time step and return core step results.

        This private method implements the actual environment transition logic. It works
        with the internal self.state dictionary that already includes actions
        and returns the core step results without observations (which are handled by the
        public step method).

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
        ### Fix current agent actions at t=0
        for ext_name, value in self.get_external_inputs().items():
            com = self.model.model.component(ext_name)
            com[com.index_set().at(1)].fix(value)  # index is 1-based

        ### Evaluate expressions
        indexed_sol, _ = self.model.pyo_get_solution(names=set(self.model.start_value_mapping.values()))
        ### Set second value of expression to corresponding parameter
        result = {param: indexed_sol[expression][1] for param, expression in self.model.start_value_mapping.items()}

        # Update model parameters with newly evaluated values, mustn't infere with expression evaluation
        self.model.pyo_update_params(result)

        # Insert values to state with correct names
        self.set_external_outputs(result)

        return 0, False, self._truncated(), {}

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the internal state of the environment and return info dictionary.

        This private method initializes the internal self.state dictionary by reading initial
        paramneter values from the PyomoModel. It does not use the seed parameter since the
        initial state is determined by the user configuration.

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

        _, scalar_sol = self.model.pyo_get_solution(names=set(self.state_config.ext_outputs))
        self.set_external_outputs(scalar_sol)

        return {}

    def close(self) -> None:
        # PyomoSimEnv doesn't need additional handling on close
        pass

    @staticmethod
    def create_state(model: pyo.ConcreteModel, model_name: str, output_dir: pathlib.Path | str | None = None) -> None:
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
