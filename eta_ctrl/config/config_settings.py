from __future__ import annotations

import math
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, GetJsonSchemaHandler, field_validator, model_validator

from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager
from eta_ctrl.util.utils import is_divisible

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic.json_schema import JsonSchemaValue

log = getLogger(__name__)


def convert_datetime(datetime_: str | datetime | None) -> datetime | None:
    """Convert a string to a datetime object using pandas."""
    if datetime_ is None or isinstance(datetime_, datetime):
        return datetime_
    return pd.to_datetime(datetime_).to_pydatetime()


class ConfigSettings(BaseModel):
    """Helper class, which is part of `Config`, for settings parameters."""

    model_config = ConfigDict(extra="allow", frozen=True, use_attribute_docstrings=True)

    seed: int | None = None
    """Seed for random sampling (default: None)."""
    verbose: int = Field(default=2, ge=0, le=3, validation_alias=AliasChoices("verbose", "verbosity"))
    """Logging verbosity of the framework (default: 2)."""
    n_environments: int = Field(default=1, ge=1)
    """Number of vectorized environments to instantiate (if not using DummyVecEnv) (default: 1)."""
    n_episodes_play: int | None = Field(default=1, ge=1)
    """Number of episodes to execute when the agent is playing (default: None)."""
    n_episodes_learn: int | None = Field(default=1, ge=1)
    """Number of episodes to execute when the agent is learning (default: None)."""
    save_model_every_x_episodes: int = Field(default=10, ge=1)
    """How often to save the model during training (default: 10 - after every ten episodes)."""
    plot_interval: int = Field(default=10, ge=1)
    """How many episodes to pass between each render call (default: 10 - after every ten episodes)."""
    scenario_time_begin: datetime | None = None
    """Beginning time of the scenario."""
    scenario_time_end: datetime | None = None
    """Ending time of the scenario."""
    use_random_time_slice: bool = False
    """Boolean flag whether to use a random time slice when the difference of
    scenario_time_end and scenario_time_begin is greater than the episode duration (default: False)."""
    sampling_time: float = Field(gt=0)
    """Duration between time samples in seconds (can be a float value)."""
    episode_duration: float = Field(gt=0)
    """Duration of an episode in seconds (can be a float value)."""
    prediction_horizon: float | None = Field(default=None, gt=0)
    """Total duration of one prediction/optimization run when used with the MPC agent."""
    sim_steps_per_sample: int | None = Field(default=None, ge=1)
    """Simulation steps for every sample."""

    scale_actions: float | None = Field(default=None)
    """Multiplier for scaling the agent actions before passing them to the environment (default: None)."""
    round_actions: int | None = Field(default=None, ge=1)
    """Number of digits to round actions to before passing them to the environment (default: None)."""

    environment: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("env", "environment", "env_specific", "environment_specific"),
    )
    """Settings dictionary for specifically the environment."""
    agent: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("agent", "agent_specific"))
    """Settings dictionary for specifically the agent."""

    log_to_file: bool = True
    """Flag which is true if the log output should be written to a file (default: True)."""

    scenario_files: list[ConfigCsvScenario] | None = None

    @property
    def n_prediction_steps(self) -> int | None:
        """Amount of steps in the prediction_horizon."""
        if self.prediction_horizon is None:
            return None
        return int(self.prediction_horizon // self.sampling_time) + 1

    def __str__(self) -> str:
        """Human-readable string representation of ConfigSettings."""
        return (
            f"ConfigSettings(episode_duration={self.episode_duration}, "
            f"sampling_time={self.sampling_time}, n_environments={self.n_environments})"
        )

    @model_validator(mode="before")
    @classmethod
    def _check_duplicate_aliases(cls, data: Any) -> Any:
        """Raise an error if both the canonical name and an alias are provided."""
        if not isinstance(data, dict):
            return data  # pragma: no cover
        env_aliases = {"env", "environment", "env_specific", "environment_specific"}
        agent_aliases = {"agent", "agent_specific"}
        for aliases, label in [(env_aliases, "environment"), (agent_aliases, "agent")]:
            found = [k for k in aliases if k in data]
            if len(found) > 1:
                msg = f"Multiple keys for '{label}' settings found: {found}. Use only '{label}'."
                raise ValueError(msg)
        return data

    @field_validator("scenario_time_begin", "scenario_time_end", mode="before")
    @classmethod
    def _convert_datetimes(cls, v: Any) -> datetime | None:
        return convert_datetime(v)

    @model_validator(mode="after")
    def _validate_time_params(self) -> ConfigSettings:
        "Check if episode duration and prediction_horizon (if not None) are a multiple"
        "of the sampling time."

        def _round_to_sampling_time(value: float | None, label: str) -> float | None:
            if value is None:
                return None
            if is_divisible(value, self.sampling_time):
                return value
            corrected = math.floor(value / self.sampling_time) * self.sampling_time
            log.warning(
                f"{label} {value} is not a multiple of sampling time {self.sampling_time}."
                f" Rounding down to {corrected}."
            )
            return corrected

        object.__setattr__(self, "episode_duration", _round_to_sampling_time(self.episode_duration, "Episode duration"))
        object.__setattr__(
            self, "prediction_horizon", _round_to_sampling_time(self.prediction_horizon, "Prediction horizon")
        )

        return self

    def model_post_init(self, _: Any) -> None:
        # Default values for environment
        self.environment.setdefault("verbose", self.verbose)
        self.environment.setdefault("sampling_time", self.sampling_time)
        self.environment.setdefault("episode_duration", self.episode_duration)
        if self.sim_steps_per_sample is not None:
            self.environment.setdefault("sim_steps_per_sample", self.sim_steps_per_sample)
        # Default values for agent
        self.agent.setdefault("seed", self.seed)
        self.agent.setdefault("verbose", self.verbose)

        if self.model_extra:
            msg = "Following values were not recognized in the config settings section and are ignored: "
            msg += ", ".join(self.model_extra.keys())
            log.warning(msg)

    def create_scenario_manager(self, scenarios_path: Path) -> None:
        """Create a ScenarioManager for the environment.

        :param scenarios_path: Path to the scenario files, default None.
        :type scenarios_path: Path
        """
        if self.scenario_files is None:
            # Don't create a scenario manager if no scenario files are given
            return

        if self.scenario_time_begin is None or self.scenario_time_end is None:
            msg = "Define scenario_time_begin and scenario_time_end in config [settings] section when using scenarios."
            raise TypeError(msg)

        scenario_timespan = (self.scenario_time_end - self.scenario_time_begin).total_seconds()
        if scenario_timespan < 0:
            msg = "scenario_time_begin must be smaller than or equal to scenario_time_end."
            raise ValueError(msg)

        duration = self.episode_duration
        # When prediction horizon is defined the duration will include it
        duration += self.prediction_horizon if self.prediction_horizon is not None else self.sampling_time

        if scenario_timespan < duration:
            msg = (
                f"Given scenario time range from {self.scenario_time_begin} to {self.scenario_time_end}"
                f" does not cover the requested duration of {duration} seconds."
            )
            raise ValueError(msg)
        scenario_configs = [
            ConfigCsvScenario(**f.model_dump(), scenarios_path=scenarios_path) for f in self.scenario_files
        ]
        self.environment["scenario_manager"] = CsvScenarioManager(
            scenario_configs=scenario_configs,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            total_time=duration,
            resample_time=self.sampling_time,
            use_random_time_slice=self.use_random_time_slice,
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        # Remove 'format': 'date-time' so the schema accepts flexible datetime strings like "2022-03-18 00:00"
        for field in ("scenario_time_begin", "scenario_time_end"):
            for entry in json_schema.get("properties", {}).get(field, {}).get("anyOf", []):
                entry.pop("format", None)
        return json_schema
