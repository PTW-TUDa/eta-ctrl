from __future__ import annotations

import csv
import json
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd
import toml
import yaml

if TYPE_CHECKING:
    from typing import Any

    from eta_ctrl.util.type_annotations import Path


log = getLogger(__name__)


def get_unique_output_path(base_path: pathlib.Path) -> pathlib.Path:
    """Get a unique output path with overwrite protection.

    This function ensures that files are not accidentally overwritten by
    appending a counter to the filename if the original path already exists.
    For example: 'file.txt' -> 'file_1.txt' -> 'file_2.txt', etc.

    :param base_path: The desired output path.
    :return: A unique path that doesn't exist.
    """
    output_path = base_path
    counter = 1
    while output_path.exists():
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        output_path = parent / f"{stem}_{counter}{suffix}"
        counter += 1
    return output_path


def json_import(path: Path) -> list[Any] | dict[str, Any]:
    """Extend standard JSON import to allow '//' comments in JSON files.

    :param path: Path to JSON file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

    try:
        # Remove comments from the JSON file (using regular expression), then parse it into a dictionary
        cleanup = re.compile(r"^((?:(?:[^\/\"])*(?:\"[^\"]*\")*(?:\/[^\/])*)*)", re.MULTILINE)
        with path.open("r") as f:
            file = "\n".join(cleanup.findall(f.read()))
        result = json.loads(file)
        log.info(f"JSON file {path} loaded successfully.")
    except OSError as e:
        log.exception(f"JSON file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise
    return result


def toml_import(path: Path) -> dict[str, Any]:
    """Import a TOML file and return the parsed dictionary.

    :param path: Path to TOML file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path)

    try:
        with path.open("r") as f:
            result = toml.load(f)
        log.info(f"TOML file {path} loaded successfully.")
    except OSError as e:
        log.exception(f"TOML file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise

    return result


def toml_export(path: Path, data: dict[str, Any]) -> None:
    """Export data to TOML file.

    :param path: Path to TOML file.
    :param data: Data to be saved as TOML.
    """
    path = pathlib.Path(path)

    try:
        with path.open("w") as f:
            toml.dump(data, f)
    except OSError as e:
        log.exception(f"TOML file couldn't be exported: {e.strerror}. Filename: {e.filename}")
        raise


def yaml_import(path: Path) -> dict[str, Any]:
    """Import a YAML file and return the parsed dictionary.

    :param path: Path to YAML file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path)

    try:
        with path.open("r") as f:
            result = yaml.safe_load(f)
        log.info(f"YAML file {path} loaded successfully.")
    except OSError as e:
        log.exception(f"YAML file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise

    return result


def csv_import(path: Path) -> dict[str, Any]:
    """Import a csv file and return the parsed dictionary.

    :param path: Path to csv file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path)

    try:
        dataframe = pd.read_csv(
            path,
            index_col=False,
            sep=";",
            decimal=".",
        )
        result = dataframe.to_dict(orient="records")
        result = {"state_vars": result}
        log.info(f"csv file {path} loaded successfully.")
    except OSError as e:
        log.exception(f"csv file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise
    return result


def load_config(file: Path) -> dict[str, Any]:
    """Load configuration from JSON, TOML, YAML or CSV file.
    The read file is expected to contain a dictionary of configuration options.
    CSV files are converted to a list of dictionaries under the key 'state_vars'.

    If `file` contains a suffix (e.g. `.csv` or `.toml`) that suffix is used
    directly. If no suffix is present the function will try all supported extensions
    (json, toml, yml, yaml, csv) in this order and pick the first matching file.

    :param file: Path to the configuration file, with or without extension.
    :return: Dictionary of configuration options.
    """
    available_importers: dict[str, Callable] = {
        ".json": json_import,
        ".toml": toml_import,
        ".yml": yaml_import,
        ".yaml": yaml_import,
        ".csv": csv_import,
    }
    config: dict[str, Any] | None = None
    file_path = pathlib.Path(file)
    suffix = file_path.suffix.lower()

    # If a suffix is provided explicitly, prefer that import method
    if suffix and suffix in available_importers:
        if file_path.exists():
            config = available_importers[suffix](file_path)
    else:
        # Try common extensions in order when no explicit suffix was provided
        for extension, import_method in available_importers.items():
            _file_path: pathlib.Path = file_path.with_suffix(extension)
            if _file_path.exists():
                config = import_method(_file_path)
                break

    if config is None:
        msg = f"Config file not found: {file}"
        raise FileNotFoundError(msg)

    if not isinstance(config, dict):
        msg = f"Config file {file} must define a dictionary of options."  # type: ignore[unreachable]
        raise TypeError(msg)

    return config


def _replace_decimal_str(value: str | float, decimal: str = ".") -> str:
    """Replace the decimal sign in a string.

    :param value: The value to replace in.
    :param decimal: New decimal sign.
    """
    return str(value).replace(".", decimal)


def csv_export(
    path: Path,
    data: Mapping[str, Any] | Sequence[Mapping[str, Any] | Any] | pd.DataFrame,
    names: Sequence[str] | None = None,
    index: Sequence[int] | pd.DatetimeIndex | None = None,
    *,
    sep: str = ";",
    decimal: str = ".",
) -> None:
    """Export data to CSV file.

    :param path: Directory path to export data.
    :param data: Data to be saved.
    :param names: Field names used when data is a Matrix without column names.
    :param index: Optional sequence to set an index
    :param sep: Separator to use between the fields.
    :param decimal: Sign to use for decimal points.
    """
    _path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    if _path.suffix != ".csv":
        _path.with_suffix(".csv")

    if isinstance(data, Mapping):
        with _path.open("a") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys(), delimiter=sep)
            if not _path.exists():
                writer.writeheader()

            writer.writerow({key: _replace_decimal_str(val, decimal) for key, val in data.items()})

    elif isinstance(data, pd.DataFrame):
        if index is not None:
            data.index = index
        data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    elif isinstance(data, Sequence):
        if names is not None:
            cols = names
        elif isinstance(data[-1], Mapping):
            cols = list(data[-1].keys())
        else:
            msg = "Column names for csv export not specified."
            raise ValueError(msg)

        _data = pd.DataFrame(data=data, columns=cols)
        if index is not None:
            _data.index = index
        _data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    log.info(f"Exported CSV data to {_path}.")
