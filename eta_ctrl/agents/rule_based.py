from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from stable_baselines3.common.base_class import BaseAlgorithm

if TYPE_CHECKING:
    import io
    import pathlib
    from typing import Any

    import torch as th
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv


log = getLogger(__name__)


class RuleBased(BaseAlgorithm, abc.ABC):
    """The rule based agent base class provides the facilities to easily build a complete rule based agent. To achieve
    this, only the *control_rules* function must be implemented. It should take an observation from the environment
    as input and provide actions as an output.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselines3.common.base_class.
    """

    def __init__(
        self,
        policy: type[BasePolicy],
        env: VecEnv,
        verbose: int = 4,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        # Ensure that arguments required by super class are always present
        super().__init__(policy=policy, env=env, verbose=verbose, learning_rate=0, **kwargs)

        #: Last / initial State of the agent.
        self.state: np.ndarray | None = np.zeros(self.action_space.shape) if self.action_space is not None else None

        self.policy_class: type[BasePolicy]
        if _init_setup_model:
            self._setup_model()

    def get_env(self) -> VecEnv:
        if self.env is None:
            msg = "Can't access attribute 'self.env', initialize environment first"
            raise AttributeError(msg)
        return self.env

    @abc.abstractmethod
    def control_rules(self, observation: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        """This function is abstract and should be used to implement control rules which determine actions from
        the received observations.

        :param observation: Observations as provided by a single, non vectorized environment.
        :return: Action values, as determined by the control rules.
        """

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Perform controller operations and return actions. It will take care of vectorization of environments.
        This will call the control_rules method which should implement the control rules for a single environment.

        :param observation: the input observation.
        :param state: The last states (not used here).
        :param episode_start: The last masks (not used here).
        :param deterministic: Whether to return deterministic actions. This agent always returns
                              deterministic actions.
        :return: Tuple of the model's action and the next state (state is typically None in this agent).
        """
        actions = []
        for idx, env_obs in enumerate(iterate(self.observation_space, observation)):
            actions.append(self.control_rules(env_obs))
            log.debug(f"Action vector for environment {idx}: {actions[idx]}")
        if not actions:
            msg = "The control_rules method must NOT return None."
            raise ValueError(msg)

        action_array = create_empty_array(self.action_space, n=self.get_env().num_envs)
        result = concatenate(self.action_space, actions, action_array)

        # Mypy can't infer types here
        if not isinstance(result, np.ndarray):
            msg = "Actions must be an np.ndarray"
            raise TypeError(msg)

        return result, None

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        _init_setup_model: bool = False,
        **kwargs: Any,
    ) -> RuleBased:
        """Load the model from a zip-file.

        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!

        :param path: path to the file (or a file-like) where to
            load the agent from.
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment.
        :param device: Device on which the code should run..
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force a call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading.
        """
        if env is None:
            msg = "Parameter env must be specified."
            raise ValueError(msg)
        model: RuleBased = super().load(path, env, device, custom_objects, print_system_info, force_reset, **kwargs)

        return model

    def _get_pretrain_placeholders(self) -> None:
        """Get tensorflow pretrain placeholders is not implemented for the rule based agent."""
        msg = "The rule based agent cannot provide tensorflow pretrain placeholders."
        raise NotImplementedError(msg)

    def get_parameter_list(self) -> None:
        """Get tensorflow parameters is not implemented for the rule based agent."""
        msg = "The rule based agent cannot provide a tensorflow parameter list."
        raise NotImplementedError(msg)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> RuleBased:
        """Return a trained model. Learning is not implemented for the rule based agent.

        :param total_timesteps: The total number of samples (env steps) to train on.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: The name of the run for TensorBoard logging.
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging).
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: The trained model.
        """
        return self

    def _setup_model(self) -> None:
        if self.policy_class is not None:
            self.policy: type[BasePolicy] = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
            )

    def action_probability(
        self,
        observation: dict[str, np.ndarray],
        state: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Get the model's action probability distribution from an observation. This is not implemented for this
        agent.
        """
        msg = "The rule based agent cannot calculate action probabilities."
        raise NotImplementedError(msg)
