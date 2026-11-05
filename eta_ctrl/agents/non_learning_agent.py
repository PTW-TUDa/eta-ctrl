from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING, Self

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm

from eta_ctrl.common.sb3_extensions.policies import NoPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv

log = getLogger(__name__)


class NonLearningAgent(BaseAlgorithm, abc.ABC):
    """Base class for agents which do not learn and determine their actions by name (e.g. rule based or optimization
    based agents).

    This class provides a mapping from action names to the action array via
    *actions_order*, as well as the boilerplate needed to use non-learning agents in place of stable_baselines3
    agents within ETA Ctrl (see :py:meth:`eta_ctrl.core.EtaCtrl.play`).

    The agents are only partially compatible with the stable_baselines3 interface: Although the class inherits from
    :py:class:`~stable_baselines3.common.base_class.BaseAlgorithm`, there is no policy (the *policy* attribute is not
    set) and no learned state. Only :py:meth:`predict` is functional. :py:meth:`learn` only logs an error and returns
    the agent, while :py:meth:`save`, :py:meth:`load` and :py:meth:`get_parameters` raise
    :py:class:`NotImplementedError`. Do not rely on other stable_baselines3 features (e.g. the policy or callbacks).

    :param policy: Agent policy. Parameter is not used in these agents and ignored.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param actions_order: Ordered names of the actions the agent controls (usually defined by the StateConfig).
                          They determine the order of the action array. This is set automatically in the agent settings
                          by :py:class:`~eta_ctrl.config.config.Config`.
    :param kwargs: Additional arguments as specified in stable_baselines3.common.base_class.
    """

    def __init__(
        self,
        policy: type[BasePolicy] | None,
        env: VecEnv,
        verbose: int = 1,
        *,
        actions_order: list[str],
        **kwargs: Any,
    ) -> None:
        # Ensure that arguments required by super class are always present
        super().__init__(
            policy=NoPolicy,
            env=env,
            verbose=verbose,
            learning_rate=0.0,
            supported_action_spaces=(spaces.Box,),
            **kwargs,
        )

        #: Ordered names of the controlled actions. They define the order of the action array.
        self.actions_order: list[str] = actions_order

    def get_env(self) -> VecEnv:
        """Helper method for type annotation."""
        if self.env is None:
            msg = "Can't access attribute 'self.env', initialize environment first"
            raise AttributeError(msg)
        return self.env

    def _setup_model(self) -> None:
        """Required method by the BaseAlgorithm interface. Non-learning agents have no policy, so there is nothing
        to set up.
        """

    def _actions_to_array(self, action_dict: Mapping[str, float | np.ndarray]) -> np.ndarray:
        """Convert a dictionary of named actions to an action array in the order of *actions_order*.

        :param action_dict: Action values by action name.
        :return: Flat action array for a single environment.
        """
        unknown = set(action_dict) - set(self.actions_order)
        missing = [name for name in self.actions_order if action_dict.get(name) is None]
        if unknown or missing:
            msg = f"Invalid actions. Missing: {missing}. Unknown: {unknown}."
            raise ValueError(msg)

        return np.concatenate(
            [np.atleast_1d(np.asarray(action_dict[name], dtype=np.float32)) for name in self.actions_order]
        )

    @abc.abstractmethod
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Determine the actions for the given observations (by the agent's decision logic).

        `state`, `episode_start` and `deterministic` exist for SB3 compatibility and are usually ignored.

        :param observation: The observations of all (vectorized) environments.
        :param state: The last states (not used here).
        :param episode_start: The last masks (not used here).
        :param deterministic: Whether to return deterministic actions. These agents always return
                                   deterministic actions.
        :return: `(actions, state)` with `actions` matching `self.action_space` for each environment.
        """

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Self:
        """Not supported: these agents determine their actions without learning. Logs an error and returns the agent
        unchanged.

        :param total_timesteps: The total number of samples (env steps) to train on.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: The name of the run for TensorBoard logging.
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging).
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: The agent.
        """
        log.error(f"{type(self).__name__} cannot learn, skipping learn(). Use it for playing only.")
        return self

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Not supported: these agents hold no learned state to persist."""
        msg = f"{type(self).__name__} cannot be saved; non-learning agents have no learned state."
        raise NotImplementedError(msg)

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Self:
        """Not supported: these agents hold no learned state to load."""
        msg = f"{cls.__name__} cannot be loaded; non-learning agents have no learned state. Remove the model file."
        raise NotImplementedError(msg)

    def get_parameters(self) -> dict[str, dict]:
        """Not supported: these agents have no policy and no learned parameters."""
        msg = f"{type(self).__name__} has no parameters; non-learning agents have no policy."
        raise NotImplementedError(msg)
