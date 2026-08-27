from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import pyomo.environ as pyo
from gymnasium import spaces
from gymnasium.vector.utils import create_empty_array, iterate
from pyomo import opt
from pyomo.common.errors import InfeasibleConstraintException
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from eta_ctrl.common.sb3_extensions.policies import NoPolicy
from eta_ctrl.simulators import PyomoModel
from eta_ctrl.util.utils import import_class_from_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy as np
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import MaybeCallback

log = getLogger(__name__)


class MpcAgent(BaseAlgorithm):
    """Simple, Pyomo based optimization agent supporting multiple solvers.

    The MpcAgent requires a PyomoModel which is passed via the `model_import` parameter.
    It must be defined in the config under the 'agent_specific' section.
    Common stablebaselines3 parameters are ignored for the MpcAgent as it cannot be used for learning or training.
    It can only be used to predict actions via predict(); use EtaCtrl.play() to run experiments.

    :param policy: Agent policy. Parameter is not used in this agent
    :param env: Environment to be optimized
    :param sampling_time: Interval for one timestep. Used to calcucate n_prediction_steps
    :param prediction_horizon: Duration of the prediction in seconds (usually a subsample of the episode duration)
    :param model_import: Dotted import path to the PyomoModel subclass (e.g. ``"my_package.my_module.MyModel"``)
    :param verbose: Logging verbosity
    :param solver_name: Name of the solver (e.g. gurobi, cplex, or glpk). Is passed to ``pyomo.SolverFactory``.
    :param action_index: Index of the solution value to be used as action
        (by default the value for the first timestep in the solution will be used)
    :param model_parameters: Dictionary of parameters to forward to the PyomoModel
    :param solver_options: Dictionary of solver options (e.g. time limits or tolerances)
    :param solver_callback: Optional callback function called after each solve step
    """

    def __init__(
        self,
        env: VecEnv,
        sampling_time: float,
        prediction_horizon: float,
        model_import: str,
        verbose: int = 1,
        *,
        solver_name: str = "cplex",
        action_index: int = 0,
        model_parameters: dict[str, Any] | None = None,
        solver_options: dict[str, Any] | None = None,
        solver_callback: Callable[[BaseAlgorithm], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=NoPolicy,
            env=env,
            learning_rate=0.0,
            verbose=verbose,
            use_sde=False,
            monitor_wrapper=False,
            supported_action_spaces=(spaces.Box,),
        )
        log.setLevel(int(verbose * 10))  # Set logging verbosity

        if isinstance(self.get_env(), VecNormalize):
            msg = "The MPC agent does not allow the use of normalized environments."
            raise TypeError(msg)

        #: Specification of the order in which action values should be returned.
        self.actions_order = self.get_env().get_attr("state_config", 0)[0].actions

        # Solver parameters
        #: Name of the solver to be used
        self.solver_name: str = solver_name
        #: Index of the solution value to be used as action (if this is 0, the first value in a list
        #: of solution values will be used).
        self.action_index = action_index
        #: Additional callback for predicting
        self.solver_callback = solver_callback

        #: Pyomo solver instance
        self.solver = pyo.SolverFactory(self.solver_name)
        self.solver.options.update(solver_options or {})  # Adjust solver settings

        self.policy_class: type[BasePolicy]

        target_class: type[PyomoModel] = import_class_from_module(model_import, base_class=PyomoModel)
        #: PyomoModel instance
        self.model: PyomoModel = target_class(
            model_parameters=model_parameters,
            sampling_time=sampling_time,
            prediction_horizon=prediction_horizon,
        )
        # Shortcut for model access
        self.concrete_model: pyo.ConcreteModel = self.model.model

        self._setup_model()

    def _setup_model(self) -> None:
        """Required method by the BaseAlgorithm interface."""
        if self.policy_class is not None:
            self.policy: type[BasePolicy] = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
            )

    def get_env(self) -> VecEnv:
        """Helper method for type annotation."""
        if self.env is None:
            msg = "Can't access attribute 'self.env', initialize environment first"
            raise AttributeError(msg)
        return self.env

    def solve(self) -> pyo.ConcreteModel:
        """Solve the current pyomo model instance with given parameters. This could also be used separately to solve
        normal MILP problems. Since the entire problem instance is returned, result handling can be outsourced.

        :return: Solved pyomo model instance.
        """

        _tee: bool = bool(log.level / 10 <= 1)

        result = self.solver.solve(self.model.model, symbolic_solver_labels=True, tee=_tee)

        if _tee:
            print("\n")  # noqa: T201 (print is ok here, because cplex prints directly to console).
        log.debug(
            "Problem information:\n%s\n%s\n%s",
            "\t+----------------------------------+",
            "\n".join(
                f"\t {item}: {value.value} "
                for item, value in result["Problem"][0].items()
                if not isinstance(value.value, opt.UndefinedData)
            ),
            "\t+----------------------------------+",
        )

        # Log status after the optimization
        log.info(
            "Solver information:\n%s\n%s\n%s",
            "\t+----------------------------------+",
            "\n".join(
                f"\t {item}: {value.value} "
                for item, value in result["Solver"][0].items()
                if item != "Statistics" and not isinstance(value.value, opt.UndefinedData)
            ),
            "\t+----------------------------------+",
        )

        # Log status after the optimization
        if len(result["Solution"]) >= 1:
            log.debug(
                "Solution information:\n%s\n%s\n\t%s",
                "\t+----------------------------------+",
                "\n".join(
                    f"\t {item}: {value.value} "
                    for item, value in result["Solution"][0].items()
                    if not isinstance(value.value, opt.UndefinedData)
                ),
                "\t+----------------------------------+",
            )

        # Check if no optimal solution could be found
        if not opt.check_optimal_termination(result):
            self.handle_solve_failed(result=result)

        return self.model.model

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Solve the current pyomo model instance with given parameters and observations and return the optimal actions.

        :param observation: the input observation (not used here).
        :param state: The last states (not used here).
        :param episode_start: The last masks (not used here).
        :param deterministic: Whether to return deterministic actions. This agent always returns
                                   deterministic actions.
        :return: Tuple of the model's action and the next state (not used here).
        """
        action_array: np.ndarray = create_empty_array(self.action_space, n=self.get_env().num_envs)  # type: ignore[assignment]

        # Return actions for each environment
        for idx, env_obs in enumerate(iterate(self.observation_space, observation)):
            env_obs_: dict = env_obs  # for typing only, must be of type dictionary

            # Update model parameters with environment observations
            self.model.pyo_update_params(env_obs_)

            # Solve the model for actions
            self.solve()

            if self.solver_callback:
                self.solver_callback(self)

            # Aggregate the agent actions from pyomo component objects
            solution = {}
            for com in self.model.model.component_objects(pyo.Var):
                com = cast("pyo.Var", com)
                if isinstance(com, pyo.ScalarVar):
                    continue
                try:
                    solution[com.name] = pyo.value(com[com.index_set().at(self.action_index + 1)])  # index is 1-based
                except (ValueError, KeyError) as e:
                    model_name = type(self.model).__name__
                    msg = f"Couldn't fetch the value for action '{com.name}' in the PyomoModel {model_name}"
                    raise ValueError(msg) from e

            for i, action in enumerate(self.actions_order):
                log.debug(f"Action '{action}' value: {solution[action]}")
                action_array[idx][i] = solution[action]

        return action_array, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Self:
        """The MPC approach cannot learn a new model.
        Specify the model attribute as a pyomo Concrete model instead, to use the prediction function of this agent.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: The trained model.
        """
        return self

    def handle_solve_failed(self, result: Any) -> None:
        """Called when the solver did not reach an optimal solution.

        If a feasible (suboptimal) solution exists, logs a warning and returns so the caller
        can continue with that solution. If no feasible solution exists, logs full diagnostics
        and raises an :exc:`InfeasibleConstraintException`.

        :param result: Result object returned by the Pyomo solver.
        """
        if len(result["Solution"]) != 0:
            gap_info = ""
            if "Gap" in result["Solution"][0]:
                gap_value = result["Solution"][0]["Gap"].value
                if not isinstance(gap_value, opt.UndefinedData):
                    gap_info = f" (achieved MIP gap: {gap_value})"
            log.warning(
                "Solver did not reach optimal solution%s. "
                "Termination condition: %s, Status: %s. "
                "Continuing with best available solution.",
                gap_info,
                result.solver.termination_condition,
                result.solver.status,
            )
            return

        log.error(
            "Solver failed: no feasible solution found. Termination condition: %s, Status: %s. %s",
            result.solver.termination_condition,
            result.solver.status,
            f"Message: {getattr(result.solver, 'message', '')}",
        )

        result_str = str(result)
        if len(result_str) > 10000:
            try:
                log_dir = Path(self.get_env().get_attr("run_info", 0)[0].results_path)
                result_file = log_dir / f"solver_result_failure_{self.num_timesteps}.txt"
                result_file.write_text(result_str, encoding="utf-8")
                log.debug("Full solver result saved to: %s", result_file)
            except (OSError, AttributeError, IndexError, TypeError) as e:
                log.warning("Could not save result to disk: %s. Logging truncated version.", e)
                log.debug("Solver result (truncated): %s", result_str[:5000] + "...")
        else:
            log.debug("Full solver result: %s", result)

        msg = "Solver failed to find feasible solution."
        raise InfeasibleConstraintException(msg)
