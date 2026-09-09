from __future__ import annotations

import warnings
from logging import getLogger
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

log = getLogger(__name__)

_DEPRECATED_PATHS_KEYS = frozenset({"monitor_filename"})


class ConfigPaths(BaseModel):
    """Relative paths and filenames set in the experiment config ``paths`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_keys(cls, data: Any) -> Any:
        """Warn when a deprecated ``paths`` key is still specified in the config."""
        if "monitor_filename" in data:
            warnings.warn(
                "'monitor_filename' is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data

    state_file_relpath: Path | None = None
    """Relative path to the state_config file (default: [environment_classname]_state_config).
    The method :py:meth:`~eta_ctrl.envs.StateConfig.from_file` will first try at the root path,
    then search in the /environment folder if not successful.
    """

    results_relpath: Path = Path("results")
    """Relative path to the results folder."""

    scenarios_relpath: Path = Path("scenarios")
    """Relative path to the scenarios folder."""

    models_relpath: Path = Path("models")
    """Relative path to model checkpoints within the series results folder."""

    model_filename: Path = Path("model.zip")
    """Filename of the saved ML model."""

    model_before_error_filename: Path = Path("model_before_error.zip")
    """Filename used to preserve a ML model when learning fails."""

    info_filename: Path = Path("info.json")
    """Filename of the run information."""

    monitor_filename: Path = Path("monitor.csv")
    """Filename of the monitoring data."""

    log_output_filename: Path = Path("log_output.log")
    """Filename of the run log."""

    vec_normalize_filename: Path = Path("vec_normalize.pkl")
    """Filename of the SB3 normalization wrapper."""

    net_arch_filename: Path = Path("net_arch.txt")
    """Filename of the neural network architecture."""
