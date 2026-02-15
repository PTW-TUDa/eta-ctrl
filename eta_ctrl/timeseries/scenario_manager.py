from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path  # noqa: TC003, pydantic needs this for type validation at runtime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from eta_ctrl import timeseries
from eta_ctrl.util.type_annotations import FillMethod, InferDatetimeType  # noqa: TC001

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any

    import pandas as pd


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
    @abstractmethod
    def get_scenario_state(self, n_steps: int, columns: list[str] | None = None) -> dict[str, np.ndarray]:
        """Get scenario values for the current time step.

        :param n_steps: Current time step.
        :param columns: Optional list of column names to return. If None, returns all columns.
        :return: Dictionary with scenario data for the requested columns.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scenario_state_with_duration(
        self, n_step: int, duration: int, columns: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Get scenario values for the interval [n_step, n_step+duration].

        :param n_step: Current time step.
        :param duration: Additional amount of steps in interval.
        :param columns: Optional list of column names to return. If None, returns all columns.
        :return: Dictionary with scenario data for the requested columns.
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
        resample_time: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._data: pd.DataFrame

        self.scenario_configs: list[ConfigCsvScenario] = scenario_configs
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.resample_time = resample_time

        self.seed = seed

        self.load_data()

    def load_data(self) -> None:
        """Load scenario data by calling 'scenario_from_csv' with the ConfigCsvScenario objects"""
        random = np.random.default_rng(self.seed) if self.seed is not None else None
        self.scenarios = timeseries.scenario_from_csv(
            scenario_configs=self.scenario_configs,
            start_time=self.start_time,
            end_time=self.end_time,
            total_time=self.total_time,
            random=random,
            resample_time=self.resample_time,
            prefix_renamed=True,
        )
        self.total_length = len(self.scenarios)

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

    def get_scenario_state(self, n_steps: int, columns: list[str] | None = None) -> dict[str, np.ndarray]:
        if n_steps >= self.total_length:
            msg = f"n_steps {n_steps} is out of bounds for scenarios with length {len(self.scenarios)}"
            raise IndexError(msg)

        valid_columns = self._validate_columns(columns)
        vals = self.scenarios.loc[self.scenarios.index[n_steps], valid_columns]
        if isinstance(vals, (int, float)):
            # Single column case - vals is a scalar
            return {valid_columns[0]: np.array([vals])}
        return {name: np.array([val]) for name, val in vals.to_dict().items()}

    def get_scenario_state_with_duration(
        self, n_step: int, duration: int, columns: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        end_index = n_step + duration

        if end_index > self.total_length:
            msg = (
                f"Requested data from {n_step} to {end_index} ({duration} steps) "
                f"but only {self.total_length} steps available. "
                f"Shortfall: {end_index - self.total_length} steps."
            )
            raise IndexError(msg)

        valid_columns = self._validate_columns(columns)
        return self.scenarios.loc[self.scenarios.index[n_step:end_index], valid_columns].to_dict()
