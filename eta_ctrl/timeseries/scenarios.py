from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from eta_ctrl import timeseries

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario
    from eta_ctrl.util.type_annotations import TimeStep


def import_scenario_config(
    scenario_config: ConfigCsvScenario,
    prefix_renamed: bool,
    slice_begin: datetime,
    slice_end: datetime,
    resample_time: TimeStep,
) -> pd.DataFrame:
    """Load a DataFrame from a ConfigCsvScenario object.

    :param scenario_config: Config for csv file.
    :type scenario_config: ConfigCsvScenario
    :param prefix_renamed: Whether newly prefixed values should get renamed too
    :type prefix_renamed: bool
    :param start_time: Starting time for the scenario import
    :type slice_begin: datetime
    :param end_time: Latest ending time for the scenario import
    :type slice_end: datetime
    :param resample_time: Resample the scenario data to the specified interval. If given as an int, this will be
        interpreted as seconds
    :type resample_time: TimeStep
    :raises ValueError: When none of the csv columns span from slice_begin to slice_end
    :return: DataFrame with desired datetime index.
    :rtype: pd.DataFrame
    """
    data = timeseries.df_from_csv(
        scenario_config.abs_path,
        infer_datetime_from=scenario_config.infer_datetime_cols,
        time_conversion_str=scenario_config.time_conversion_str,
    )

    # Resample before slicing to possibly get more context
    data = timeseries.df_resample(data, resample_time, missing_data=scenario_config.interpolation_method)
    data = data[slice_begin:slice_end].copy()  # type: ignore[misc]
    scaling = scenario_config.scale_factors
    col_names = {}
    for col in data.columns:
        col_names[col] = _fix_col_name(
            name=col,
            prefix=scenario_config.prefix,
            prefix_renamed=prefix_renamed,
            rename_cols=scenario_config.rename_cols,
        )
        # Apply scaling before renaming
        if scaling is not None and col in scaling:
            data[col] = data[col].multiply(scaling[col])

    # Make sure that the resulting file corresponds to the requested time slice
    if len(data) <= 0 or data.first_valid_index() > slice_begin or data.last_valid_index() < slice_end:
        msg = (
            f"The loaded scenario file {scenario_config.abs_path.name} does not contain enough data for the entire "
            "selected time slice. Or the set scenario times do not correspond to the provided data."
        )
        raise ValueError(msg)

    # rename all columns with the name mapping determined above
    return data.rename(columns=col_names)


def scenario_from_csv(
    scenario_configs: list[ConfigCsvScenario],
    *,
    start_time: datetime,
    end_time: datetime,
    resample_time: TimeStep,
    prefix_renamed: bool = True,
) -> pd.DataFrame:
    """Import (possibly multiple) scenario data files from csv files and return them as a single pandas
    data frame. The import function supports column renaming and will slice and resample data as specified.

    :raises ValueError: If start and/or end times are outside the scope of the imported scenario files.

    :param start_time: Starting time for the scenario import.
    :param end_time: Latest ending time for the scenario import.
    :param resample_time: Resample the scenario data to the specified interval. If given as an int, this will be
                        interpreted as seconds.
    :param prefix_renamed: Should prefixes be applied to renamed columns as well?
                           When setting this to false make sure that all columns in all loaded scenario files
                           have different names. Otherwise, there is a risk of overwriting data.
    :return: Imported and processed data as pandas.DataFrame.
    """

    scenarios = []
    for scenario_config in scenario_configs:
        data = import_scenario_config(
            scenario_config=scenario_config,
            prefix_renamed=prefix_renamed,
            resample_time=resample_time,
            slice_begin=start_time,
            slice_end=end_time,
        )
        scenarios.append(data)
    return pd.concat(scenarios, axis=1)


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
