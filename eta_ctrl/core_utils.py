from __future__ import annotations

import inspect
import pathlib
from functools import partial
from typing import TYPE_CHECKING

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymnasium import Env
    from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

    from eta_ctrl.config import ConfigRun
    from eta_ctrl.envs import BaseEnv
    from eta_ctrl.util.type_annotations import AlgoSettings, EnvSettings, Path
from logging import getLogger

log = getLogger(__name__)


def vectorize_environment(
    env: type[BaseEnv],
    config_run: ConfigRun,
    env_settings: EnvSettings,
    callback: Callable[[BaseEnv], None],
    verbose: int = 2,
    vectorizer: type[DummyVecEnv] = DummyVecEnv,
    n: int = 1,
    *,
    training: bool = False,
    monitor_wrapper: bool = False,
    norm_wrapper_obs: bool = False,
    norm_wrapper_reward: bool = False,
) -> VecNormalize | VecEnv:
    """Vectorize the environment and automatically apply normalization wrappers if configured. If the environment
    is initialized as an interaction_env it will not have normalization wrappers and use the appropriate configuration
    automatically.

    :param env: Environment class which will be instantiated and vectorized.
    :param config_run: Configuration for a specific optimization run.
    :param env_settings: Configuration settings dictionary for the environment which is being initialized.
    :param callback: Callback to call with an environment instance.
    :param verbose: Logging verbosity to use in the environment.
    :param vectorizer: Vectorizer class to use for vectorizing the environments.
    :param n: Number of vectorized environments to create.
    :param training: Flag to identify whether the environment should be initialized for training or playing. If true,
                     it will be initialized for training.
    :param norm_wrapper_obs: Flag to determine whether observations from the environments should be normalized.
    :param norm_wrapper_reward: Flag to determine whether rewards from the environments should be normalized.
    :return: Vectorized environments, possibly also wrapped in a normalizer.
    """
    # Create the vectorized environment
    log.debug("Trying to vectorize the environment.")
    # Ensure n is one, if the DummyVecEnv is used (it doesn't support more than one)
    if vectorizer.__class__.__name__ == "DummyVecEnv" and n != 1:
        n = 1
        log.warning("Setting number of environments to 1 because DummyVecEnv (default) is used.")

    if "verbose" in env_settings and env_settings["verbose"] is not None:
        verbose = env_settings.pop("verbose")

    # Create the vectorized environment
    def create_env(env_id: int) -> Env:
        env_id += 1
        return env(env_id=env_id, config_run=config_run, verbose=verbose, callback=callback, **env_settings)

    envs: VecEnv | VecNormalize
    envs = vectorizer([partial(create_env, i) for i in range(n)])

    # The VecMonitor knows the ep_reward and so this can be logged to tensorboard
    if monitor_wrapper:
        envs = VecMonitor(envs)

    # Automatically normalize the input features
    if norm_wrapper_obs or norm_wrapper_reward:
        # check if normalization data is available and load it if possible, otherwise
        # create a new normalization wrapper.
        if config_run.path_vec_normalize.is_file():
            log.info(
                f"Normalization data detected. Loading running averages into normalization wrapper: \n"
                f"\t {config_run.path_vec_normalize}"
            )
            envs = VecNormalize.load(str(config_run.path_vec_normalize), envs)
            envs.training = training
            envs.norm_obs = norm_wrapper_obs
            envs.norm_reward = norm_wrapper_reward
        else:
            log.info("No Normalization data detected.")
            envs = VecNormalize(envs, training=training, norm_obs=norm_wrapper_obs, norm_reward=norm_wrapper_reward)

    return envs


def _check_tensorboard_log(tensorboard_log: bool, log_path: Path | None) -> dict[str, str]:
    """Create necessary arguments for tensorboard logging if required

    :param tensorboard_log: Flag to enable logging to tensorboard.
    :param log_path: Path for tensorboard log. Online required if logging is true
    :return: Kwargs for the agent.
    """
    if tensorboard_log:
        if log_path is None:
            msg = "If tensorboard logging is enabled, a path for results must be specified as well."
            raise ValueError(msg)
        _log_path = pathlib.Path(log_path)
        log.info(f"Tensorboard logging is enabled. Log file: {_log_path}")
        log.info(
            f"Please run the following command in the console to start tensorboard: \n"
            f'tensorboard --logdir "{_log_path}" --port 6006'
        )
        return {"tensorboard_log": str(_log_path)}
    return {}


def initialize_model(
    algo: type[BaseAlgorithm],
    policy: type[BasePolicy],
    envs: VecEnv | VecNormalize,
    algo_settings: AlgoSettings,
    seed: int | None = None,
    *,
    tensorboard_log: bool = False,
    log_path: Path | None = None,
) -> BaseAlgorithm:
    """Initialize a new model or algorithm.

    :param algo: Algorithm to initialize.
    :param policy: The policy that should be used by the algorithm.
    :param envs: The environment which the algorithm operates on.
    :param algo_settings: Additional settings for the algorithm.
    :param seed: Random seed to be used by the algorithm.
    :param tensorboard_log: Flag to enable logging to tensorboard.
    :param log_path: Path for tensorboard log. Online required if logging is true
    :return: Initialized model.
    """
    log.debug(f"Trying to initialize model: {algo.__name__}")

    # tensorboard logging
    algo_kwargs = _check_tensorboard_log(tensorboard_log, log_path)

    # check if the agent takes all the default parameters.
    algo_settings.setdefault("seed", seed)

    algo_params = inspect.signature(algo).parameters
    if "seed" not in algo_params and inspect.Parameter.VAR_KEYWORD not in {p.kind for p in algo_params.values()}:
        del algo_settings["seed"]
        log.warning(
            f"'seed' is not a valid parameter for agent {algo.__name__}. This default parameter will be ignored."
        )

    # create model instance
    return algo(policy, envs, **algo_settings, **algo_kwargs)  # type: ignore[arg-type]


def load_model(
    algo: type[BaseAlgorithm],
    envs: VecEnv | VecNormalize,
    algo_settings: AlgoSettings,
    path_model: Path,
    *,
    tensorboard_log: bool = False,
    log_path: Path | None = None,
) -> BaseAlgorithm:
    """Load an existing model.

    :param algo: Algorithm type of the model to be loaded.
    :param envs: The environment which the algorithm operates on.
    :param algo_settings: Additional settings for the algorithm.
    :param path_model: Path to load the model from.
    :param tensorboard_log: Flag to enable logging to tensorboard.
    :param log_path: Path for tensorboard log. Online required if logging is true
    :return: Initialized model.
    """
    log.debug(f"Trying to load existing model: {path_model}")
    _path_model = path_model if isinstance(path_model, pathlib.Path) else pathlib.Path(path_model)

    if not _path_model.exists():
        msg = f"Model couldn't be loaded. Path not found: {_path_model}"
        raise OSError(msg)

    # tensorboard logging
    algo_kwargs = _check_tensorboard_log(tensorboard_log, log_path)

    try:
        model = algo.load(_path_model, envs, **algo_settings, **algo_kwargs)  # type: ignore[arg-type]
        log.debug("Model loaded successfully.")
    except OSError as e:
        msg = f"Model couldn't be loaded: {e.strerror}. Filename: {e.filename}"
        raise OSError(msg) from e

    return model
