from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003, pydantic needs this for type validation at runtime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from eta_ctrl import timeseries
from eta_ctrl.util.type_annotations import FillMethod, InferDatetimeType  # noqa: TC001

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any

    import numpy as np
    import pandas as pd

    from eta_ctrl.envs.state import StateVar

log = logging.getLogger(__name__)


class ConfigCsvScenario(BaseModel):
    #: :meta private:
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Relative path to the scenario
    path: str

    #: Method of interpolation
    interpolation_method: FillMethod | None = None
    #: Scale factors for each columns
    scale_factors: dict[str, float] | None = None
    #: Prefix for all column names
    prefix: str | None = None
    #: Setting how the datetime values should be converted.
    #: When set to string it uses the format from ``time_conversion_str``,
    #: when set to 'dates' it will use pandas to determine the datetime.
    #: If a two-tuple (row, col) is given, data from the specified field in the data files
    #: will be used to determine the date format
    infer_datetime_cols: InferDatetimeType | tuple[int, int] = "dates"
    #: Time conversion string used when ``infer_datetime_cols`` is set to 'string'
    #: Should specify the format for Python ``strptime``
    time_conversion_str: str = "%Y-%m-%d %H:%M"
    #: Dictionary for renaming column names
    #:
    #:  .. note::
    #:
    #:    The column names are stripped of illegal characters and underscores are added in place of spaces.
    #:    "Water Temperature #2 [°C]" becomes "Water_Temperature_2_C". If you want to rename the column,
    #:    you need to specify the processed name, for example: {"Water_Temperature_2_C": "T_W"}.
    rename_cols: dict[str, str] | None = None

    #: Directory for the scenarios.
    #: Not included in config declaration, passed by main Config object
    scenarios_path: Path = Field(exclude=True)

    def model_post_init(self, _: Any) -> None:
        """Ensure that the CSV file exists.

        :raises FileNotFoundError: If file does not exist.
        """
        if not self.abs_path.exists():
            msg = "Scenario file does not exist"
            raise FileNotFoundError(msg)

    @property
    def abs_path(self) -> Path:
        """Absolute file path of the scenario."""
        return (self.scenarios_path / self.path).resolve()


class ScenarioManager(ABC):
    def compute_episode_offset(self, rng: np.random.Generator) -> int:
        """Compute the row offset into the scenario data for the next episode.

        Returns 0 by default (no random slicing). Override in subclasses that support
        random time slicing.

        :param rng: Random number generator from the environment.
        :return: Integer row offset into the scenario data.
        """
        return 0

    def get_scenario_state_var(self, n_step: int, state_var: StateVar) -> np.ndarray:
        """Get scenario values for a single state variable at the given (absolute) step.

        :param n_step: Absolute row index into the scenario data (env step + episode offset).
        :param state_var: State variable configuration.
        :return: Array of scenario values.
        """
        scenario_id = state_var.scenario_id

        data = self._get_data(n_step=n_step, names=[scenario_id])  # type: ignore[list-item]
        return data[scenario_id]  # type: ignore[index]

    @abstractmethod
    def _get_data(self, n_step: int, duration: int = 1, names: list[str] | None = None) -> dict[str, np.ndarray]:
        """Get scenario values for the interval [n_step, n_step+duration].

        :param n_step: Absolute row index into the scenario data (env step + episode offset).
        :param duration: Number of steps to retrieve.
        :param names: Column names to retrieve. If None, all columns are returned.
        :return: Dictionary mapping column names to value arrays.
        """
        raise NotImplementedError


class CsvScenarioManager(ScenarioManager):
    """ScenarioManager class for loading scenario data from CSV files."""

    def __init__(
        self,
        scenario_configs: list[ConfigCsvScenario],
        start_time: datetime,
        end_time: datetime,
        total_time: float,
        resample_time: float,
        use_random_time_slice: bool = False,
    ) -> None:
        super().__init__()
        self._data: pd.DataFrame

        self.scenario_steps = int(total_time / resample_time)

        self.scenario_configs: list[ConfigCsvScenario] = scenario_configs
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.resample_time = resample_time
        self.use_random_time_slice = use_random_time_slice

        self.load_data()

    def compute_episode_offset(self, rng: np.random.Generator) -> int:
        """Compute the row offset into the scenario dataframe for the next episode.

        :param rng: Random number generator used to pick a random starting position.
        :return: Integer row index into self.scenarios representing the episode start.
        """
        if not self.use_random_time_slice:
            return 0
        available_space = self.total_df_length - self.scenario_steps
        if available_space == 0:
            return 0
        return rng.choice(range(available_space)).item()

    def load_data(self) -> None:
        """Load scenario data by calling 'scenario_from_csv' with the ConfigCsvScenario objects"""
        self.scenarios = timeseries.scenario_from_csv(
            scenario_configs=self.scenario_configs,
            start_time=self.start_time,
            end_time=self.end_time,
            resample_time=self.resample_time,
            prefix_renamed=True,
        )
        self.total_df_length = len(self.scenarios)

    def _validate_columns(self, columns: list[str] | None) -> list[str]:
        """Validate and return the list of columns to retrieve.

        :param columns: Requested column names, or None for all columns.
        :return: List of valid column names to retrieve.
        :raises KeyError: If any requested column is not found in the scenario data.
        """
        if columns is None:
            return list(self.scenarios.columns)

        missing_cols = set(columns) - set(self.scenarios.columns)
        if missing_cols:
            available_cols = list(self.scenarios.columns)
            msg = (
                f"Requested scenario columns {sorted(missing_cols)} not found in loaded scenario data. "
                f"Available columns: {available_cols}"
            )
            raise KeyError(msg)

        return columns

    def _get_data(self, n_step: int, duration: int = 1, names: list[str] | None = None) -> dict[str, np.ndarray]:
        end_index = n_step + duration

        if end_index > self.total_df_length:
            msg = (
                f"Requested data from {n_step} to {end_index} ({duration} steps) "
                f"but only {self.total_df_length} steps available. "
                f"Shortfall: {end_index - self.total_df_length} steps."
            )
            raise IndexError(msg)
        # Choose all columns if names are not supplied
        columns = self._validate_columns(columns=names)
        return {col: self.scenarios.iloc[n_step:end_index][col].to_numpy() for col in columns}
