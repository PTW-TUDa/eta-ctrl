from __future__ import annotations

import abc
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyomo.environ as pyo
from gymnasium import spaces
from gymnasium.vector.utils import create_empty_array, iterate
from pyomo import opt
from pyomo.common.errors import InfeasibleConstraintException
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from typing_extensions import Self

from eta_ctrl.common.sb3_extensions.policies import NoPolicy
from eta_ctrl.simulators import PyomoModel

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import MaybeCallback

log = getLogger(__name__)


class MpcAgent(BaseAlgorithm):
    """Simple, Pyomo based optimization agent supporting multiple solvers.

    The agent requires an environment that specifies the 'model' attribute, returning a
    :py:class:`pyomo.ConcreteModel` and a sorted list as the order for the action space. This list is used to
    avoid ambiguity when returning a list of actions. Since the model specifies its own action and observation
    space, this agent does not use the *action_space* and *observation_space* specified by the environment.

    :param policy: Agent policy. Parameter is not used in this agent.
    :param env: Environment to be optimized.
    :param verbose: Logging verbosity.
    :param solver_name: Name of the solver, could be cplex or glpk.
    :param action_index: Index of the solution value to be used as action (if this is 0, the first value in a list
        of solution values will be used).
    :param kwargs: Additional arguments as specified in stable_baselines3.common.base_class or as provided by solver.
    """

    @property
    @abc.abstractmethod
    def model_file(self) -> Path | str:
        """Relative path to the MPC model."""
        return ""

    def __init__(
        self,
        env: VecEnv,
        sampling_time: float,
        prediction_horizon: float,
        verbose: int = 1,
        *,
        policy: type[BasePolicy] | None = None,
        solver_name: str = "cplex",
        action_index: int = 0,
        **kwargs: Any,
    ) -> None:
        # Prepare kwargs to be sent to the super class and to the solver.
        super_args: dict[str, Any] = {}

        # Set default values for superclass arguments
        super_args = {"supported_action_spaces": (spaces.Box,), "monitor_wrapper": False}
        super_args["seed"] = kwargs.pop("seed", None)
        super_args.setdefault("learning_rate", 0.0)

        for unused_kwarg in (
            "policy_base",
            "learning_rate",
            "policy_kwargs",
            "device",
            "support_multi_env",
            "create_eval_env",
            "use_sde",
            "sde_sample_freq",
        ):
            kwargs.pop(unused_kwarg, None)

        super().__init__(policy=NoPolicy, env=env, verbose=verbose, **super_args)
        log.setLevel(int(verbose * 10))  # Set logging verbosity

        if isinstance(self.get_env(), VecNormalize):
            msg = "The MPC agent does not allow the use of normalized environments."
            raise TypeError(msg)

        #: Specification of the order in which action values should be returned.
        self.actions_order = self.get_env().get_attr("state_config", 0)[0].actions

        self.policy_class: type[BasePolicy]

        #: Index of the solution value to be used as action (if this is 0, the first value in a list
        #: of solution values will be used).
        self.action_index = action_index

        model_path = (self.get_env().get_attr("path_env", 0)[0] / self.model_file).resolve()
        target_class = PyomoModel.import_mpc_class(model_path)

        self.model: PyomoModel = target_class(
            model_parameters=kwargs.pop("model_parameters"),
            sampling_time=sampling_time,
            prediction_horizon=prediction_horizon,
        )
        #: Pyomo optimization model as specified by the environment.
        self.concrete_model: pyo.ConcreteModel = self.model.model

        # Solver parameters
        self.solver_name: str = solver_name
        self.solver = pyo.SolverFactory(self.solver_name)
        self.solver.options.update(kwargs)  # Adjust solver settings

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
        if (
            result.solver.termination_condition != opt.TerminationCondition.optimal
            or result.solver.status != opt.SolverStatus.ok
        ):
            # Log warning with achieved gap information if available
            gap_info = ""
            if len(result["Solution"]) >= 1 and "Gap" in result["Solution"][0]:
                gap_value = result["Solution"][0]["Gap"].value
                if not isinstance(gap_value, opt.UndefinedData):
                    gap_info = f" (achieved MIP gap: {gap_value})"

            log.warning(
                f"Solver did not reach optimal solution within constraints{gap_info}. "
                f"Termination condition: {result.solver.termination_condition}, "
                f"Status: {result.solver.status}. Continuing with best available solution."
            )

            # Check if there is at least a feasible solution to work with
            if len(result["Solution"]) == 0 or result.solver.termination_condition in {
                opt.TerminationCondition.infeasible,
                opt.TerminationCondition.invalidProblem,
                opt.TerminationCondition.solverFailure,
                opt.TerminationCondition.internalSolverError,
                opt.TerminationCondition.error,
            }:
                # Log detailed diagnostic information for debugging
                log.error(
                    "Problem has no feasible solution. Solver details:\n"
                    "  Termination condition: %s\n"
                    "  Solver status: %s\n"
                    "  Number of solutions: %d\n"
                    "  Solver message: %s",
                    result.solver.termination_condition,
                    result.solver.status,
                    len(result["Solution"]),
                    result.solver.message if hasattr(result.solver, "message") else "N/A",
                )

                # Log full result object - save to disk if too large
                result_str = str(result)
                if len(result_str) > 10000:  # If result is larger than 10KB
                    # Save to disk instead of cluttering logs
                    try:
                        log_dir = Path(self.get_env().get_attr("config_run", 0)[0].results_path)
                        result_file = log_dir / f"solver_result_failure_{self.num_timesteps}.txt"
                        result_file.write_text(result_str, encoding="utf-8")
                        log.debug("Full solver result saved to: %s", result_file)
                    except (OSError, AttributeError, IndexError, TypeError) as e:
                        log.warning("Could not save result to disk: %s. Logging truncated version.", e)
                        log.debug("Solver result (truncated): %s", result_str[:5000] + "...")
                else:
                    # Small enough to log directly
                    log.debug("Full solver result object: %s", result)

                self.get_env().env_method("handle_failed_solve", self.model, result)

                # Raise appropriate exception instead of sys.exit
                msg = (
                    f"Solver failed to find feasible solution. "
                    f"Termination condition: {result.solver.termination_condition}, "
                    f"Status: {result.solver.status}"
                )
                raise InfeasibleConstraintException(msg)

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
