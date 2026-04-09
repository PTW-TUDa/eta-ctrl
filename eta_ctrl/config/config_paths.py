from __future__ import annotations

from logging import getLogger
from pathlib import Path

from pydantic import BaseModel, ConfigDict

log = getLogger(__name__)


class ConfigPaths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    state_file_relpath: Path | None = None
    """Relative path to the state_config file (default: [environment_classname]_state_config).
    The method :py:meth:`~eta_ctrl.envs.StateConfig.from_file` will first try at the root path,
    then search in the /environment folder if not successful.
    """

    results_relpath: Path = Path("results")
    """Relative path to the results folder."""

    scenarios_relpath: Path = Path("scenarios")
    """Relative path to the scenarios folder."""
