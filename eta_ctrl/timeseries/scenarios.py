from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from eta_ctrl import timeseries
from eta_ctrl.util.utils import timestep_to_timedelta

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    import numpy as np

    from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario
    from eta_ctrl.util.type_annotations import TimeStep


def scenario_from_csv(
    scenario_configs: list[ConfigCsvScenario],
    *,
    start_time: datetime,
    end_time: datetime | None = None,
    total_time: TimeStep | None = None,
    random: np.random.Generator | bool | None = False,
    resample_time: TimeStep | None = None,
    prefix_renamed: bool = True,
) -> pd.DataFrame:
    """Import (possibly multiple) scenario data files from csv files and return them as a single pandas
    data frame. The import function supports column renaming and will slice and resample data as specified.

    :raises ValueError: If start and/or end times are outside the scope of the imported scenario files.

    .. note::
        The ValueError will only be raised when this is true for all files. If only one file is outside
        the range, an empty series will be returned for that file.

    :param start_time: Starting time for the scenario import.
    :param end_time: Latest ending time for the scenario import (default: inferred from start_time and total_time).
    :param total_time: Total duration of the imported scenario. If given as int this will be
                       interpreted as seconds (default: inferred from start_time and end_time).
    :param random: Set to true if a random starting point (within the interval determined by
                   start_time and end_time) should be chosen. This will use the environments' random generator.
    :param resample_time: Resample the scenario data to the specified interval. If given as an int, this will be
                        interpreted as seconds. If resample_time is None, it will be treated as 0 (default: None).
    :param prefix_renamed: Should prefixes be applied to renamed columns as well?
                           When setting this to false make sure that all columns in all loaded scenario files
                           have different names. Otherwise, there is a risk of overwriting data.
    :return: Imported and processed data as pandas.DataFrame.
    """

    # Set defaults and convert values where necessary
    if total_time is not None:
        total_time = timestep_to_timedelta(total_time)

    # If resample_time is None, default to 0
    resample_time = resample_time if resample_time is not None else 0
    _resample_time = timestep_to_timedelta(resample_time)

    _random = random if random is not None else False

    slice_begin, slice_end = timeseries.find_time_slice(
        start_time,
        end_time,
        total_time=total_time,
        random=_random,
        round_to_interval=_resample_time,
    )

    def import_scenario(
        scenario_config: ConfigCsvScenario,
    ) -> pd.DataFrame:
        data = timeseries.df_from_csv(
            scenario_config.abs_path,
            infer_datetime_from=scenario_config.infer_datetime_cols,
            time_conversion_str=scenario_config.time_conversion_str,
        )
        data = timeseries.df_resample(data, _resample_time, missing_data=scenario_config.interpolation_method)
        data = data[slice_begin:slice_end].copy()  # type: ignore[misc]
        col_names = {}
        for col in data.columns:
            col_names[col] = _fix_col_name(
                name=col,
                prefix=scenario_config.prefix,
                prefix_renamed=prefix_renamed,
                rename_cols=scenario_config.rename_cols,
            )
            scaling = scenario_config.scale_factors
            if scaling is None:
                continue
            if col in scaling:
                data[col] = data[col].multiply(scaling[col])

        # rename all columns with the name mapping determined above
        return data.rename(columns=col_names)

    scenario = pd.DataFrame()
    for scenario_config in scenario_configs:
        data = import_scenario(scenario_config=scenario_config)
        scenario = pd.concat((data, scenario), axis=1)

    # Make sure that the resulting file corresponds to the requested time slice
    if (
        len(scenario) <= 0
        or scenario.first_valid_index() > slice_begin + _resample_time
        or scenario.last_valid_index() < slice_end - _resample_time
    ):
        msg = (
            "The loaded scenario file does not contain enough data for the entire selected time slice. Or the set "
            "scenario times do not correspond to the provided data."
        )
        raise ValueError(msg)

    return scenario


def _fix_col_name(
    name: str,
    *,
    prefix: str | None = None,
    prefix_renamed: bool = False,
    rename_cols: Mapping[str, str] | None = None,
) -> str:
    """Figure out correct name for the column.

    :param name: Name to rename.
    :param prefix: Prefix to prepend to the name.
    :param prefix_renamed: Prepend prefix if name is renamed?
    :param rename_cols: Mapping of old names to new names.
    """
    rename_cols = rename_cols if rename_cols is not None else {}

    # Keep the same name if no new name is provided
    new_name = str(rename_cols.get(name, name))

    if prefix is None:
        return new_name

    # Prefix is given but should not be applied
    if name != new_name and not prefix_renamed:
        return new_name

    return f"{prefix}_{new_name}"
