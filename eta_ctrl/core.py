from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from eta_ctrl.common import (
    CallbackEnvironment,
    log_net_arch,
    log_run_config,
    log_to_file,
    merge_callbacks,
)
from eta_ctrl.config import Config, RunInfo

from .core_utils import (
    initialize_model,
    load_model,
    vectorize_environment,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from typing import Any

    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.type_aliases import MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv, VecNormalize
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

    from eta_ctrl.util.type_annotations import Path

log = getLogger(__name__)


class EtaCtrl:
    """Set up an experiment from a configuration file and execute optimization runs.

    *EtaCtrl* holds the (immutable) experiment configuration and knows how to build everything a single
    optimization run needs: a :class:`~eta_ctrl.config.RunInfo`, the vectorized environments and the agent.
    Executing a run is the responsibility of :class:`~eta_ctrl.core.EtaCtrlRun`, which :meth:`prepare_run` creates.
    The :meth:`learn` and :meth:`play` methods combine both steps and are the usual entry points.

    :param config_name: Name of configuration file in configuration directory (should be JSON format).
    :param root_path: Root path of the application (the configuration will be interpreted relative to this).
    :param config_overwrite: Dictionary to overwrite selected configurations.
    :param config_relpath: Path to configuration file, relative to root path.
    """

    def __init__(
        self,
        config_name: str,
        root_path: Path | None = None,
        config_overwrite: Mapping[str, Any] | None = None,
        config_relpath: Path | None = None,
    ) -> None:
        #: Configuration of the experiment (immutable, shared by all runs of this experiment).
        self.config: Config = Config.from_file(
            root_path=root_path, config_relpath=config_relpath, config_name=config_name, overwrite=config_overwrite
        )
        log.setLevel(int(self.config.settings.verbose * 10))

    def __str__(self) -> str:
        """Human-readable string representation of EtaCtrl."""
        env_class = self.config.setup.environment_class.__name__
        agent_class = self.config.setup.agent_class.__name__
        return f"EtaCtrl(config='{self.config.config_name}', env={env_class}, agent={agent_class})"

    def __repr__(self) -> str:
        """Developer-friendly string representation of EtaCtrl."""
        return f"EtaCtrl(config_name='{self.config.config_name}', root_path='{self.config.root_path}')"

    @contextmanager
    def prepare_run(
        self,
        *,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        reset: bool = False,
        training: bool = False,
    ) -> Generator[EtaCtrlRun, None, None]:
        """Prepare a single optimization run and provide it as a context manager.

        Creating the run info (including its results folders), the environments and the agent happens on entry;
        the environments are closed again when the context exits. Use this if you need access to the prepared
        environments or the agent. :meth:`learn` and :meth:`play` are shortcuts around it.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :param reset: Flag to determine whether the model should reinitialize its parameters.
              If true, any existing model files are renamed to timestamped
              ``.bak`` files (see :func:`~eta_ctrl.util.io_utils.backup_file`) and new ones are
              initialized. If false, existing files are loaded.
        :param training: Should preparation be done for training/learning. Set this to True if you
                         intend to call :meth:`~eta_ctrl.core.EtaCtrlRun.learn`.
        :return: Context manager yielding the prepared :class:`~eta_ctrl.core.EtaCtrlRun`.
        """
        run_info = self._create_run_info(
            series_name if series_name is not None else "",
            run_name if run_name is not None else "",
            run_description,
        )

        with self.prepare_environments(run_info=run_info, training=training) as environments:
            model = self.prepare_model(environments=environments, run_info=run_info, reset=reset)
            yield EtaCtrlRun(config=self.config, run_info=run_info, model=model, environments=environments)

    def _create_run_info(self, series_name: str, run_name: str, run_description: str = "") -> RunInfo:
        """Create the run info of a single run, create its results folders and start logging to file.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :return: Run info of the prepared run.
        """
        run_info = RunInfo(
            series=series_name,
            name=run_name,
            description=run_description,
            root_path=self.config.root_path,
            paths=self.config.paths,
        )
        run_info.create_results_folders()

        # Add file handler to parent logger to log the terminal output
        log_to_file(config=self.config, run_info=run_info)

        log.info("Run prepared successfully.")

        return run_info

    def prepare_model(
        self, *, environments: VecEnv | VecNormalize, run_info: RunInfo, reset: bool = False
    ) -> BaseAlgorithm:
        """Check for an existing model and load it or back it up and create a new model.

        :param environments: Vectorized environments the model should be attached to.
        :param run_info: Run info of the run the model belongs to.
        :param reset: Flag to determine whether an existing model should be reset.
        :return: Loaded or newly initialized model.
        """
        model_path = run_info.run_model_path
        if model_path.is_file() and reset:
            log.info(f"Existing model detected: {model_path}")

            bak_name = model_path / f"_{datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y%m%d_%H%M')}.bak"
            model_path.rename(bak_name)
            log.info(f"Reset is active. Existing model will be backed up. Backup file name: {bak_name}")
        elif model_path.is_file():
            log.info(f"Existing model detected: {model_path}. Loading model.")

            return load_model(
                self.config.setup.agent_class,
                environments,
                self.config.settings.agent,
                run_info.run_model_path,
                tensorboard_log=self.config.setup.tensorboard_log,
                log_path=run_info.series_results_path,
            )

        # Initialize the model if it wasn't loaded from a file
        return initialize_model(
            self.config.setup.agent_class,
            self.config.setup.policy_class,
            environments,
            self.config.settings.agent,
            self.config.settings.seed,
            tensorboard_log=self.config.setup.tensorboard_log,
            log_path=run_info.series_results_path,
        )

    @contextmanager
    def prepare_environments(
        self, *, run_info: RunInfo, training: bool = True
    ) -> Generator[VecEnv | VecNormalize, None, None]:
        """Context manager which creates the environments and closes them after it exits.

        :param run_info: Run info of the run the environments belong to.
        :param training: Should preparation be done for training/learning
        """
        environments = None
        try:
            environments = self._prepare_environments(run_info=run_info, training=training)
            yield environments
        finally:
            if environments is None:
                log.error("Environment initialization failed.")
            else:
                log.debug("Closing environments.")
                environments.close()

    def _prepare_environments(self, *, run_info: RunInfo, training: bool = True) -> VecEnv | VecNormalize:
        """Vectorize and prepare the environments.

        :param run_info: Run info of the run the environments belong to.
        :param training: Should preparation be done for training/learning
        :return: Vectorized environments.
        """
        env_class = self.config.setup.environment_class
        run_info.set_env_info(env_class)

        callback = CallbackEnvironment(self.config.settings.plot_interval)
        # Vectorize the environments
        return vectorize_environment(
            env=env_class,
            run_info=run_info,
            env_settings=self.config.settings.environment,
            callback=callback,
            verbose=self.config.settings.verbose,
            vectorizer=self.config.setup.vectorizer_class,
            n=self.config.settings.n_environments,
            seed=self.config.settings.seed,
            training=training,
            monitor_wrapper=self.config.setup.monitor_wrapper,
            norm_wrapper_obs=self.config.setup.norm_wrapper_obs,
            norm_wrapper_reward=self.config.setup.norm_wrapper_reward,
        )

    def learn(
        self,
        *,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        reset: bool = False,
        callbacks: MaybeCallback = None,
    ) -> None:
        """Start the learning job for an agent with the specified environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        :param callbacks: Provide additional callbacks to send to the model.learn() call.
        """
        with self.prepare_run(
            series_name=series_name, run_name=run_name, run_description=run_description, reset=reset, training=True
        ) as run:
            run.learn(callbacks=callbacks)

    def play(self, *, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        """
        with self.prepare_run(
            series_name=series_name, run_name=run_name, run_description=run_description, reset=False, training=False
        ) as run:
            run.play()


class EtaCtrlRun:
    """A single, fully prepared optimization run: run info, environments and agent belonging together.

    Instances are created by :meth:`EtaCtrl.prepare_run` and are only valid inside its context, so the
    environments are closed when the context exits.

    :param config: Configuration of the experiment this run belongs to.
    :param run_info: Run info (identity and paths) of this run.
    :param model: Agent to execute the run with.
    :param environments: Vectorized environments to execute the run in.
    """

    def __init__(
        self, config: Config, run_info: RunInfo, model: BaseAlgorithm, environments: VecEnv | VecNormalize
    ) -> None:
        #: Configuration of the experiment this run belongs to.
        self.config = config
        #: Run info (identity and paths) of this run.
        self.run_info = run_info
        #: Agent executing this run.
        self.model = model
        #: Vectorized environments of this run.
        self.environments = environments

    def __str__(self) -> str:
        """Human-readable string representation of EtaCtrlRun."""
        return f"EtaCtrlRun(series='{self.run_info.series}', name='{self.run_info.name}')"

    def __repr__(self) -> str:
        """Developer-friendly string representation of EtaCtrlRun."""
        return (
            f"EtaCtrlRun(series='{self.run_info.series}', name='{self.run_info.name}', "
            f"model={type(self.model).__name__})"
        )

    def learn(self, callbacks: MaybeCallback = None) -> None:
        """Start the learning job for the agent of this run.

        :param callbacks: Provide additional callbacks to send to the model.learn() call.
        """
        # Log some information about the model and configuration
        log_net_arch(self.model, self.run_info)
        log_run_config(self.config, self.run_info)

        # Total training length and checkpoint interval are derived from the settings.
        total_timesteps = self.config.settings.n_learn_timesteps
        save_freq = self.config.settings.checkpoint_save_freq

        # Set the seed for the environments before starting to learn
        self.environments.seed(self.config.settings.seed)

        # Create Callback for saving model checkpoints
        # Savepoint path is defined as follows (values supplied by CheckpointCallback):
        # save_path / f"{self.name_prefix}_{checkpoint_type}{num_timesteps}_steps.{extension}")
        from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: PLC0415

        callback_learn = merge_callbacks(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.run_info.models_path),
                name_prefix=self.run_info.name,
            ),
            callbacks,
        )

        # The experiments are reset before the learning phase begins, start learning
        log.info("Start learning process of agent in environment.")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_learn,
                tb_log_name=self.run_info.name,
            )
        except OSError:
            # Determine the filepath to save the model before it failed (not defined in RunInfo)
            filename = str(
                self.run_info.series_results_path
                / f"{self.run_info.name}_{self.config.paths.model_before_error_filename}"
            )
            log.info(f"Saving model to file: {filename}.")
            self.model.save(filename)
            raise

        try:
            log.debug("Resetting environment one more time to call environment callback one last time.")
            self.environments.reset()
        except ValueError as e:
            msg = "An error occurred when the environment is resetting."
            raise ValueError(msg) from e

        self._save_model()
        log.info(f"Learning finished: {self.run_info.series} / {self.run_info.name}")

    def _save_model(self) -> None:
        """Save the model and, if the environments are normalized, the normalization data."""
        log.debug(f"Saving model to file: {self.run_info.run_model_path}.")
        self.model.save(self.run_info.run_model_path)

        from stable_baselines3.common.vec_env import VecNormalize  # noqa: PLC0415

        if isinstance(self.environments, VecNormalize):
            log.debug(f"Saving environment normalization data to file: {self.run_info.vec_normalize_path}.")
            self.environments.save(str(self.run_info.vec_normalize_path))

    def play(self) -> None:
        """Play with the agent of this run in its environments."""
        # Log some information about the model and configuration
        log_net_arch(self.model, self.run_info)
        log_run_config(self.config, self.run_info)

        n_episodes_stop = self.config.settings.n_episodes_play

        # Reset the environments before starting to play
        try:
            log.debug("Resetting environments before starting to play.")
            observations = self._reset_envs()
        except Exception:
            log.exception("Resetting environments failed")
            raise
        n_episodes = 0

        log.debug("Start playing process of agent in environment.")

        _round_actions = self.config.settings.round_actions
        _scale_actions = self.config.settings.scale_actions if self.config.settings.scale_actions is not None else 1

        while n_episodes < n_episodes_stop:
            try:
                observations, dones = self._play_step(_round_actions, _scale_actions, observations)
            except BaseException as e:
                log.exception(
                    "Exception occurred during an environment step. Aborting and trying to reset environments."
                )
                try:
                    observations = self._reset_envs()
                except BaseException as followup_exception:
                    raise e from followup_exception
                log.debug("Environment reset successful - re-raising exception")
                raise

            n_episodes += sum(dones)

    def _play_step(
        self, _round_actions: int | None, _scale_actions: float, observations: VecEnvObs
    ) -> tuple[VecEnvObs, np.ndarray]:
        # set policy prediction to deterministic for playing; type: ignore
        # Type ignored because typing in eta_ctrl is bad
        action, _ = self.model.predict(observation=observations, deterministic=True)  # type: ignore[arg-type]
        # Round and scale actions if required
        if _round_actions is not None:
            action = np.round(action * _scale_actions, _round_actions)
        else:
            action *= _scale_actions

        observations, _rewards, dones, _ = self.environments.step(action)
        return observations, dones

    def _reset_envs(self) -> VecEnvObs:
        """Reset the environment before and afterwards when the play and learn function is calling.

        :return: Observations after reset.
        """
        log.debug("Resetting environments.")

        self.environments.seed(self.config.settings.seed)
        return self.environments.reset()
