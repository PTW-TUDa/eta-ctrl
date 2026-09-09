from __future__ import annotations

import pathlib  # noqa: TC003
from logging import getLogger
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationInfo, computed_field, field_validator
from typing_extensions import deprecated

from eta_ctrl.config.config_paths import ConfigPaths  # noqa: TC001

if TYPE_CHECKING:
    from typing import Any

    from eta_ctrl.envs import BaseEnv

log = getLogger(__name__)


class RunInfo(BaseModel):
    """Tracks the identity, paths, and metadata of a single ETA Ctrl run."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    series: str
    """Name of the series of runs."""
    name: str
    """Name of this run."""
    description: str = ""
    """Description of this run."""
    root_path: pathlib.Path
    """Root path of the series."""
    paths: ConfigPaths
    """Paths object used to resolve paths for this run."""

    #: Version of the environment (set via :meth:`set_env_info`).
    env_version: str | None = None
    #: Description of the environment (set via :meth:`set_env_info`).
    env_description: str | None = None

    @field_validator("series", "name")
    @classmethod
    def _validate_no_path_separators(cls, value: str, info: ValidationInfo) -> str:
        """Ensure a name can be used as a single path component on all supported operating systems."""
        if "/" in value or "\\" in value:
            msg = f"{info.field_name} must not contain path separators '/' or '\\'."
            raise ValueError(msg)
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _default_description_if_none(cls, value: Any) -> Any:
        """Treat None as an empty description."""
        return "" if value is None else value

    # Mypy does not allow decorators on top of properties, although this is official pydantic usage.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_path(self) -> pathlib.Path:
        """Absolute path to results of the run (default: root_path/results)."""
        return self.root_path / self.paths.results_relpath

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scenarios_path(self) -> pathlib.Path:
        """Absolute path to scenarios used for the run (default: root_path/scenarios)."""
        return self.root_path / self.paths.scenarios_relpath

    @computed_field  # type: ignore[prop-decorator]
    @property
    def series_results_path(self) -> pathlib.Path:
        """Absolute path to the results of the series of runs."""
        return self.results_path / self.series

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_model_path(self) -> pathlib.Path:
        """Absolute path to the model of the run."""
        return self.series_results_path / f"{self.name}_{self.paths.model_filename}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_info_path(self) -> pathlib.Path:
        """Absolute path to information about the run."""
        return self.series_results_path / f"{self.name}_{self.paths.info_filename}"

    @property
    @deprecated("RunInfo.run_monitor_path is deprecated and will be removed in a future release.")
    def run_monitor_path(self) -> pathlib.Path:
        """Deprecated: Absolute path to the monitoring information about the run."""
        return self.series_results_path / f"{self.name}_{self.paths.monitor_filename}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def vec_normalize_path(self) -> pathlib.Path:
        """Absolute path to the SB3 normalization wrapper information."""
        return self.series_results_path / self.paths.vec_normalize_filename

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_arch_path(self) -> pathlib.Path:
        """Absolute path to the neural network architecture file."""
        return self.series_results_path / self.paths.net_arch_filename

    @computed_field  # type: ignore[prop-decorator]
    @property
    def log_output_path(self) -> pathlib.Path:
        """Absolute path to the log file."""
        return self.series_results_path / f"{self.name}_{self.paths.log_output_filename}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def models_path(self) -> pathlib.Path:
        """Absolute path to model checkpoints."""
        return self.series_results_path / self.paths.models_relpath

    def __str__(self) -> str:
        """Human-readable string representation of RunInfo."""
        return f"RunInfo(series='{self.series}', name='{self.name}')"

    def __repr__(self) -> str:
        """Developer-friendly string representation of RunInfo."""
        return (
            f"RunInfo(series='{self.series}', name='{self.name}', root_path='{self.root_path}', paths='{self.paths}')"
        )

    def create_results_folders(self) -> None:
        """Create the results folders for a run (if needed)."""
        self.results_path.mkdir(parents=True, exist_ok=True)
        # Parent should be results_path, so option is not needed
        self.series_results_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)

    def set_env_info(self, env: type[BaseEnv]) -> None:
        """Set the environment information of the run to represent the given environment.
        env_* attributes default to None if this is never called.

        :param env: The environment whose description should be used.
        """
        version, description = env.get_info()
        object.__setattr__(self, "env_version", version)
        object.__setattr__(self, "env_description", description)

    @property
    def resolved_paths(self) -> dict[str, pathlib.Path]:
        """Dictionary of all paths for the run. This is for easier access and contains all
        paths of the object."""
        return {
            "root_path": self.root_path,
            "results_path": self.results_path,
            "scenarios_path": self.scenarios_path,
            "series_results_path": self.series_results_path,
            "run_model_path": self.run_model_path,
            "run_info_path": self.run_info_path,
            "vec_normalize_path": self.vec_normalize_path,
            "log_output_path": self.log_output_path,
            "models_path": self.models_path,
        }
