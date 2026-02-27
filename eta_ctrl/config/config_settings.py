from __future__ import annotations

import itertools
import math
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import Factory, converters, define, field, fields
from pandas.core.tools.datetimes import to_datetime

from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager
from eta_ctrl.util import dict_pop_any
from eta_ctrl.util.utils import is_divisible

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from attrs import Attribute


log = getLogger(__name__)


def convert_datetime(datetime_: str | datetime) -> datetime:
    """Convert a string to a datetime object using pandas."""
    if isinstance(datetime_, datetime):
        return datetime_
    return to_datetime(datetime_).to_pydatetime()


def _env_defaults(instance: ConfigSettings, attrib: Attribute, new_value: dict[str, Any] | None) -> dict[str, Any]:
    """Set default values for the environment settings."""
    _new_value = {} if new_value is None else new_value

    _new_value.setdefault("verbose", instance.verbose)
    _new_value.setdefault("sampling_time", instance.sampling_time)
    _new_value.setdefault("episode_duration", instance.episode_duration)

    if instance.sim_steps_per_sample is not None:
        _new_value.setdefault("sim_steps_per_sample", instance.sim_steps_per_sample)

    return _new_value


def _agent_defaults(instance: ConfigSettings, attrib: Attribute, new_value: dict[str, Any] | None) -> dict[str, Any]:
    """Set default values for the environment settings."""
    _new_value = {} if new_value is None else new_value

    _new_value.setdefault("seed", instance.seed)
    _new_value.setdefault("verbose", instance.verbose)

    return _new_value


@define(frozen=False, kw_only=True)
class ConfigSettings:
    #: Seed for random sampling (default: None).
    seed: int | None = field(default=None, converter=converters.optional(int))
    #: Logging verbosity of the framework (default: 2).
    verbose: int = field(
        default=2,
        converter=converters.pipe(converters.default_if_none(2), int),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: Number of vectorized environments to instantiate (if not using DummyVecEnv) (default: 1).
    n_environments: int = field(
        default=1,
        converter=converters.pipe(converters.default_if_none(1), int),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: Number of episodes to execute when the agent is playing (default: None).
    n_episodes_play: int | None = field(default=None, converter=converters.optional(int))
    #: Number of episodes to execute when the agent is learning (default: None).
    n_episodes_learn: int | None = field(default=None, converter=converters.optional(int))
    #: Flag to determine whether the interaction env is used or not (default: False).
    interact_with_env: bool = field(
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: How often to save the model during training (default: 10 - after every ten episodes).
    save_model_every_x_episodes: int = field(
        default=10,
        converter=converters.pipe(converters.default_if_none(1), int),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: How many episodes to pass between each render call (default: 10 - after every ten episodes).
    plot_interval: int = field(
        default=10,
        converter=converters.pipe(converters.default_if_none(1), int),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: Beginning time of the scenario.
    scenario_time_begin: datetime | None = field(default=None, converter=converters.optional(convert_datetime))
    #: Ending time of the scenario.
    scenario_time_end: datetime | None = field(default=None, converter=converters.optional(convert_datetime))
    #: Boolean flag whether to use a random time slice when the difference of
    #: scenario_time_end and scenario_time_begin is greater than the episode duration (default: False).
    use_random_time_slice: bool = field(default=False)
    #: Duration of an episode in seconds (can be a float value).
    episode_duration: float = field(converter=float)
    #: Duration between time samples in seconds (can be a float value).
    sampling_time: float = field(converter=float)
    #: Simulation steps for every sample.
    sim_steps_per_sample: int | None = field(default=None, converter=converters.optional(int))

    #: Multiplier for scaling the agent actions before passing them to the environment
    #: (especially useful with interaction environments) (default: None).
    scale_actions: float | None = field(default=None, converter=converters.optional(float))
    #: Number of digits to round actions to before passing them to the environment
    #: (especially useful with interaction environments) (default: None).
    round_actions: int | None = field(default=None, converter=converters.optional(int))

    #: Settings dictionary for the environment.
    environment: dict[str, Any] = field(
        default=Factory(dict),
        converter=converters.default_if_none(Factory(dict)),  # type: ignore[misc]
        on_setattr=_env_defaults,
    )  # mypy currently does not recognize converters.default_if_none
    #: Settings dictionary for the interaction environment (default: None).
    interaction_env: dict[str, Any] | None = field(default=None, on_setattr=_env_defaults)
    #: Settings dictionary for the agent.
    agent: dict[str, Any] = field(
        default=Factory(dict),
        converter=converters.default_if_none(Factory(dict)),  # type: ignore[misc]
        # mypy currently does not recognize converters.default_if_none
        on_setattr=_agent_defaults,
    )

    #: Flag which is true if the log output should be written to a file
    log_to_file: bool = field(
        default=True,
        converter=converters.pipe(converters.default_if_none(False), bool),  # type: ignore[misc]
    )

    def __attrs_post_init__(self) -> None:
        _fields = fields(ConfigSettings)
        _env_defaults(self, _fields.environment, self.environment)
        _agent_defaults(self, _fields.agent, self.agent)

        # Set standards for interaction env settings or copy settings from environment
        if self.interaction_env is not None:
            _env_defaults(self, _fields.interaction_env, self.interaction_env)
        elif self.interact_with_env is True and self.interaction_env is None:
            log.warning(
                "Interaction with an environment has been requested, but no section 'interaction_env_specific' "
                "found in settings. Reusing 'environment_specific' section."
            )
            self.interaction_env = self.environment

        if self.n_episodes_play is None and self.n_episodes_learn is None:
            msg = "At least one of 'n_episodes_play' or 'n_episodes_learn' must be specified in settings."
            raise ValueError(msg)

        if not is_divisible(self.episode_duration, self.sampling_time):
            corrected = math.floor(self.episode_duration / self.sampling_time) * self.sampling_time
            log.warning(
                f"Episode duration {self.episode_duration} is not a multiple of sampling time "
                f"{self.sampling_time}. Rounding down to {corrected}."
            )
            self.episode_duration = corrected

    @classmethod
    def from_dict(cls, dikt: dict[str, dict[str, Any]]) -> ConfigSettings:
        errors = False

        # Read general settings dictionary
        if "settings" not in dikt:
            msg = "Settings section not found in configuration. Cannot import config file."
            raise ValueError(msg)
        settings = dikt.pop("settings")

        if "seed" not in settings:
            log.info("'seed' not specified in settings, using default value 'None'")
        seed = settings.pop("seed", None)

        if "verbose" not in settings and "verbosity" not in settings:
            log.info("'verbose' or 'verbosity' not specified in settings, using default value '2'")
        verbose = dict_pop_any(settings, "verbose", "verbosity", fail=False, default=None)

        if "n_environments" not in settings:
            log.info("'n_environments' not specified in settings, using default value '1'")
        n_environments = settings.pop("n_environments", None)

        if "n_episodes_play" not in settings and "n_episodes_learn" not in settings:
            log.error("Neither 'n_episodes_play' nor 'n_episodes_learn' is specified in settings.")
            errors = True
        n_epsiodes_play = settings.pop("n_episodes_play", None)
        n_episodes_learn = settings.pop("n_episodes_learn", None)

        interact_with_env = settings.pop("interact_with_env", False)
        save_model_every_x_episodes = settings.pop("save_model_every_x_episodes", None)
        plot_interval = settings.pop("plot_interval", None)

        scenario_time_begin = settings.pop("scenario_time_begin", None)
        scenario_time_end = settings.pop("scenario_time_end", None)

        if "episode_duration" not in settings:
            log.error("'episode_duration' is not specified in settings.")
            errors = True
        episode_duration = settings.pop("episode_duration", None)

        if "sampling_time" not in settings:
            log.error("'sampling_time' is not specified in settings.")
            errors = True
        sampling_time = settings.pop("sampling_time", None)

        sim_steps_per_sample = settings.pop("sim_steps_per_sample", None)
        scale_actions = dict_pop_any(settings, "scale_interaction_actions", "scale_actions", fail=False, default=None)
        round_actions = dict_pop_any(settings, "round_interaction_actions", "round_actions", fail=False, default=None)

        if "environment_specific" not in dikt:
            log.error("'environment_specific' section not defined in settings.")
            errors = True
        environment = dikt.pop("environment_specific", None)

        if "agent_specific" not in dikt:
            log.error("'agent_specific' section not defined in settings.")
            errors = True
        agent = dikt.pop("agent_specific", None)

        interaction_env = dict_pop_any(
            dikt, "interaction_env_specific", "interaction_environment_specific", fail=False, default=None
        )

        log_to_file = settings.pop("log_to_file", False)
        use_random_time_slice: bool = settings.pop("use_random_time_slice", False)

        # Log configuration values which were not recognized.
        for name in itertools.chain(settings, dikt):
            log.warning(
                f"Specified configuration value '{name}' in the settings section of the configuration "
                f"was not recognized and is ignored."
            )

        if errors:
            msg = "Not all required values were found in settings (see log). Could not load config file."
            raise ValueError(msg)

        return cls(
            seed=seed,
            verbose=verbose,
            n_environments=n_environments,
            n_episodes_play=n_epsiodes_play,
            n_episodes_learn=n_episodes_learn,
            interact_with_env=interact_with_env,
            save_model_every_x_episodes=save_model_every_x_episodes,
            plot_interval=plot_interval,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            use_random_time_slice=use_random_time_slice,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            sim_steps_per_sample=sim_steps_per_sample,
            scale_actions=scale_actions,
            round_actions=round_actions,
            environment=environment,
            agent=agent,
            interaction_env=interaction_env,
            log_to_file=log_to_file,
        )

    def create_scenario_manager(self, scenarios_path: Path | None = None) -> None:
        """Create a ScenarioManager for the environment.

        :param scenarios_path: Path to the scenario files, default None.
        :type scenarios_path: Path
        """
        raw_configs: list[dict[str, Any]] | None = self.environment.get("scenario_files")
        if raw_configs is None:
            # Don't create a scenario manager if no scenario files are given
            return

        if scenarios_path is None:
            msg = "Define scenarios_path in config [settings] section when using scenarios."
            raise TypeError(msg)

        if self.scenario_time_begin is None or self.scenario_time_end is None:
            msg = "Define scenario_time_begin and scenario_time_end in config [settings] section when using scenarios."
            raise TypeError(msg)

        if self.scenario_time_begin > self.scenario_time_end:
            msg = "scenario_time_begin must be smaller than or equal to scenario_time_end."
            raise ValueError(msg)

        # When prediction horizon is defined the duration will include it
        if prediction_horizon := self.environment.get("prediction_horizon"):
            try:
                prediction_horizon = float(prediction_horizon)
            except ValueError:
                log.exception("Prediction horizon needs to be defined as a number.")
                raise
            duration = self.episode_duration + prediction_horizon
        else:
            duration = self.episode_duration + self.sampling_time

        if (self.scenario_time_end - self.scenario_time_begin).total_seconds() < duration:
            msg = (
                f"Given scenario time range from {self.scenario_time_begin} to {self.scenario_time_end}"
                f"does not cover the requested duration of {duration} seconds."
            )
            raise ValueError(msg)
        scenario_configs = [
            ConfigCsvScenario(**raw_config, scenarios_path=scenarios_path) for raw_config in raw_configs
        ]
        self.environment["scenario_manager"] = CsvScenarioManager(
            scenario_configs=scenario_configs,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            total_time=duration,
            resample_time=self.sampling_time,
            use_random_time_slice=self.use_random_time_slice,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            msg = f"The key {name} does not exist - it cannot be set."
            raise KeyError(msg)
        setattr(self, name, value)
