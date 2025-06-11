from __future__ import annotations

import itertools
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import Factory, converters, define, field, fields

from eta_ctrl.util import dict_pop_any

if TYPE_CHECKING:
    from typing import Any

    from attrs import Attribute


log = getLogger(__name__)


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

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            msg = f"The key {name} does not exist - it cannot be set."
            raise KeyError(msg)
        setattr(self, name, value)
