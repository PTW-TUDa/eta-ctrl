"""Simple helpers for reading timeseries data from a csv file and getting slices or resampled data.
This module handles data using pandas dataframe objects.
"""

from __future__ import annotations

import csv
import operator as op
import pathlib
import re
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, get_args

import pandas as pd

from eta_ctrl.util.type_annotations import FillMethod
from eta_ctrl.util.utils import timestep_to_seconds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from eta_ctrl.util.type_annotations import InferDatetimeType, Path, TimeStep

log = getLogger(__name__)


def df_from_csv(
    path: Path,
    *,
    delimiter: str = ";",
    infer_datetime_from: InferDatetimeType | Sequence[int] | tuple[int, int] = "dates",
    time_conversion_str: str = "%Y-%m-%d %H:%M",
) -> pd.DataFrame:
    """Take data from a csv file, process it and return a Timeseries (pandas Data Frame) object.

    Open and read the .csv file, perform error checks and ensure that valid float values are obtained. This
    assumes that the first column is always the date and time column and provides multiple methods to convert
    this column. It also assumes that the first row is the header row.
    The header row is converted to lower case and spaces are converted to _. If header values contain special
    characters, everything starting from the first special character is discarded.

    :param path: Path to the .csv file.
    :param delimiter: Delimiter used between csv fields.
    :param infer_datetime_from: Specify how date and time values should be inferred. This can be 'dates' or 'string'
                                or a tuple/list with two values.

                                * If 'dates' is specified, pandas will be used to automatically infer the datetime
                                  format from the file.
                                * If 'string' is specified, the parameter 'time_conversion_str' must specify the
                                  string (in python strptime format) to convert datetime values.
                                * If a tuple/list of two values is given, the time format specification (according to
                                  python strptime format) will be read from the specified field in the .csv
                                  file ('row', 'column').

    :param time_conversion_str: Time conversion string according to the python (strptime) format.
    """
    path = pathlib.Path(path)

    conversion_string = None
    infer_datetime_format = False
    parse_dates = False
    if isinstance(infer_datetime_from, str):
        infer_datetime_format = infer_datetime_from == "dates"
        conversion_string = time_conversion_str if infer_datetime_from == "string" else None
    elif isinstance(infer_datetime_from, list | tuple):
        if len(infer_datetime_from) != 2:
            msg = f"Field for date format must be specified in the format ['row', 'col']. Got {infer_datetime_from}"
            raise ValueError(msg)

    else:
        msg = (
            "infer_datetime_from must be one of 'dates', 'string', or a tuple of ('row', 'col'), "
            f"Got: {infer_datetime_from}"
        )
        raise TypeError(msg)

    # Read names from header and format them such that they can be used easily as dataframe indices.
    # If required by infer_datetime_from also read time format from the file.
    with path.open("r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first_line = next(reader)
            if isinstance(infer_datetime_from, list | tuple):
                if infer_datetime_from[0] > 0:
                    for _ in range(1, infer_datetime_from[0]):
                        conversion_line = next(reader)
                else:
                    conversion_line = first_line
                conversion_string = "%" + conversion_line[infer_datetime_from[1]].split("%", 1)[1].strip()
        except StopIteration:
            msg = (
                f"The CSV file does not contain the specified date format field {infer_datetime_from}. \n"
                f"File path: {path}"
            )
            raise EOFError(msg) from None

        # Find number of fields, names of fields and a conversion string for time
        length = len(first_line)
        splitter = re.compile("[^A-Za-z0-9 _-]")
        names = [splitter.sub("", s.strip()).strip().replace(" ", "_") for s in first_line]

    # Load the CSV file
    if infer_datetime_format:
        parse_dates = True

    def converter(val: str) -> float:
        val = str(val).strip().replace(" ", "").replace(",", ".")
        return float(val) if len(val) > 0 else float("nan")

    data = pd.read_csv(
        path,
        header=0,
        names=names,
        delimiter=delimiter,
        index_col=0,
        parse_dates=parse_dates,
        converters=dict.fromkeys(range(1, length), converter),
    )

    if conversion_string:
        data.index = pd.to_datetime(data.index, format=conversion_string)

    log.info(f"Loaded data from csv file: {path}")
    return data


def round_datetime_to_interval(dt: datetime, interval: float) -> datetime:
    """Round a datetime object down to match the given interval.

    :param dt: Datetime object to round
    :type dt: datetime
    :param interval: Resampling interval
    :type interval: float
    :return: Rounded datetime object
    :rtype: datetime
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if interval == 0:
        return dt
    return datetime.fromtimestamp((dt.timestamp() // interval) * interval)


def df_resample(
    dataframe: pd.DataFrame, *periods_deltas: TimeStep, missing_data: FillMethod | None = None
) -> pd.DataFrame:
    """Resample the time index of a data frame.

    This method can be used for resampling in multiple different periods
    with multiple different deltas between single time entries.

    :param df: DataFrame for processing.
    :param periods_deltas: If one argument is specified, this will resample the data to the specified interval
                           in seconds. If more than one argument is specified, they will be interpreted as
                           (periods, interval) pairs. The first argument specifies a number of periods that should
                           be resampled, the second value specifies the interval that these periods should be
                           resampled to. A third argument would determine the next number of periods that should
                           be resampled to the interval specified by the fourth argument and so on.
    :param missing_data: Specify a method for handling missing data values. If this is not specified, missing
                         data will not be handled. Valid methods are: 'ffill', 'bfill', 'interpolate', 'asfreq'.
                         Default is 'asfreq'.
    :return: Resampled copy of the DataFrame.
    """
    # Set default value for missing_data
    missing_data = missing_data or "asfreq"

    valid_methods = get_args(FillMethod)
    if missing_data not in valid_methods:
        msg = f"Invalid value for 'missing_data': {missing_data}. Valid values are: {valid_methods}"
        raise ValueError(msg)

    interpolation_method = op.methodcaller(missing_data)

    if not dataframe.index.is_unique:
        duplicates = dataframe.index.duplicated(keep="first")
        log.warning(f"Index has non-unique values. Dropping duplicates: {dataframe.index[duplicates].to_list()}.")
        dataframe = dataframe[~duplicates]

    _periods_deltas: list[int] = [int(timestep_to_seconds(t)) for t in periods_deltas]

    if len(_periods_deltas) == 1:
        delta = _periods_deltas[0]
        new_df = (
            df_interpolate(dataframe, delta)  # interpolate needs an extra method for handling indices
            if missing_data == "interpolate"
            else interpolation_method(dataframe.resample(f"{int(delta)}s"))
        )
    else:
        period_delta = [(_periods_deltas[i], _periods_deltas[i + 1]) for i in range(0, len(_periods_deltas), 2)]
        total_periods: int = 0
        new_df = None
        for period, delta in period_delta:
            period_df = dataframe.iloc[total_periods : total_periods + period + 1]
            resampled_df = df_resample(period_df, delta, missing_data=missing_data)
            new_df = resampled_df if new_df is None else pd.concat((new_df, resampled_df))

            total_periods += period

    # Remove the duplicate value between periods
    new_df = new_df[~new_df.index.duplicated(keep="first")]

    if new_df.isna().to_numpy().any():
        log.warning(
            "Resampled Dataframe has missing values. Before using this data, ensure you deal with the missing values. "
            "For example, you could interpolate(), ffill() or dropna()."
        )

    return new_df


def df_interpolate(
    dataframe: pd.DataFrame, freq: TimeStep, limit_direction: Literal["both", "forward", "backward"] = "both"
) -> pd.DataFrame:
    """Interpolate missing values in a DataFrame with a specified frequency.
    Is able to handle unevenly spaced time series data.

    :param dataframe: DataFrame for interpolation.
    :param freq: Frequency of the resulting DataFrame.
    :param limit_direction: Direction in which to limit the
            interpolation. Defaults to "both".

    :return: Interpolated DataFrame.
    """
    if not dataframe.index.is_unique:
        log.warning(
            f"Index has non-unique values. Dropping duplicates: "
            f"{dataframe.index[dataframe.index.duplicated(keep='first')].to_list()}."
        )
        dataframe = dataframe[~dataframe.index.duplicated(keep="first")]

    freq_seconds: float = timestep_to_seconds(freq)
    freq_str: str = str(int(freq_seconds)) + "s"

    old_index: pd.Index = dataframe.index
    start: pd.Timestamp = old_index.min().floor(freq_str)
    end: pd.Timestamp = old_index.max().ceil(freq_str)
    new_index: pd.Index = pd.date_range(start=start, end=end, freq=freq_str)
    tmp_index: pd.Index = old_index.union(new_index)

    df_reindexed: pd.DataFrame = dataframe.reindex(tmp_index)
    df_interpolated: pd.DataFrame = df_reindexed.interpolate(method="time", limit_direction=limit_direction)
    new_df: pd.DataFrame = df_interpolated.reindex(new_index)

    if new_df.iloc[0].isna().to_numpy().any():
        log.warning("The first value of the interpolated dataframe is NaN.")
    if new_df.iloc[-1].isna().to_numpy().any():
        log.warning("The last value of the interpolated dataframe is NaN.")

    return new_df
