from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler

import __main__

# Pydantic needs type annotations at runtime, ruff doesn't know.
from eta_ctrl.config.config_paths import ConfigPaths
from eta_ctrl.config.config_settings import ConfigSettings  # noqa: TC001
from eta_ctrl.config.config_setup import ConfigSetup  # noqa: TC001
from eta_ctrl.envs.state import StateConfig
from eta_ctrl.util.io_utils import load_config
from eta_ctrl.util.utils import camel_to_snake_case, deep_mapping_update

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic.json_schema import JsonSchemaValue

    from eta_ctrl.util.type_annotations import Path as StrPath

log = getLogger(__name__)


class Config(BaseModel):
    """Configuration for the optimization, which can be loaded from a JSON, TOML, or YAML file.

    Holds the config name and path configuration, remaining data is split up in `ConfigSetup` and `ConfigSettings`.
    Should be instantiated via the .from_file() method.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)

    root_path: Path = Field(exclude=True)
    """Root folder path for the optimization run (default: parent folder of invoking script).
    Default value is only set when creating config via a file."""

    config_file_relpath: Path = Field(exclude=True)

    paths: ConfigPaths = Field(default_factory=ConfigPaths)
    """Optimization run paths."""

    setup: ConfigSetup
    """Optimization run setup."""
    settings: ConfigSettings
    """Optimization run settings."""

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        # Remove fields that are provided programmatically (not from the config file)
        for field in ("root_path", "config_file_relpath"):
            json_schema.get("properties", {}).pop(field, None)
            if "required" in json_schema:
                json_schema["required"] = [r for r in json_schema["required"] if r != field]
        return json_schema

    @property
    def config_name(self) -> str:
        """Name of the config file."""
        return self.config_file_relpath.name

    @property
    def results_path(self) -> Path:
        """Path to the results folder (default: root_path/results)."""
        return self.root_path / self.paths.results_relpath

    @property
    def scenarios_path(self) -> Path:
        """Path to the scenarios folder (default: root_path/scenarios)."""
        return self.root_path / self.paths.scenarios_relpath

    def __str__(self) -> str:
        """Human-readable string representation of Config."""
        return (
            f"Config '{self.config_name}' "
            f"(env={self.setup.environment_class.__name__}, agent={self.setup.agent_class.__name__})"
        )

    def model_post_init(self, _: Any) -> None:
        # MpcAgent needs sampling_time and prediction_horizon
        from eta_ctrl.agents.mpc_agent import MpcAgent  # noqa: PLC0415

        if issubclass(self.setup.agent_class, MpcAgent):
            self.settings.agent["sampling_time"] = self.settings.sampling_time
            self.settings.agent["prediction_horizon"] = self.settings.prediction_horizon

        # Create StateConfig (moved to helper to lower function complexity)
        self._create_state_config()
        self.settings.create_scenario_manager(self.scenarios_path)

    def _create_state_config(self) -> None:
        env_name = self.setup.environment_class.__name__
        # set default state file path based on environment class name if not provided in config
        state_file_relpath = self.paths.state_file_relpath or camel_to_snake_case(env_name) + "_state_config"

        # If prediction_horizon is set, we need to include n_prediction_steps as parameter
        extra_params = {}
        if self.settings.n_prediction_steps is not None:
            extra_params["n_prediction_steps"] = self.settings.n_prediction_steps

        state_config = StateConfig.from_file(
            root_path=self.root_path, filename=state_file_relpath, extra_params=extra_params
        )

        # Pass to ConfigSettings environment section
        self.settings.environment["state_config"] = state_config
        state_file_relpath = (
            state_config.source_file.relative_to(self.root_path) if state_config.source_file is not None else Path()
        )
        # Finally set correct state config location (with file ending)
        object.__setattr__(self.paths, "state_file_relpath", state_file_relpath)

    @classmethod
    def from_file(
        cls,
        config_name: str,
        root_path: StrPath | None = None,
        config_relpath: StrPath | None = None,
        overwrite: Mapping[str, Any] | None = None,
    ) -> Config:
        if root_path is None:
            # Use parent folder of invoking script when root_path is not provided
            root_path = Path(__main__.__file__).parent.resolve()
        elif not isinstance(root_path, Path):
            root_path = Path(root_path)

        if config_relpath is None:
            config_relpath = "config"

        # Load file content
        config_file_relpath = Path(config_relpath) / config_name
        file_path = root_path / config_file_relpath
        config = load_config(file_path)

        if overwrite is not None:
            config = deep_mapping_update(config, overwrite)

        return Config(root_path=root_path, config_file_relpath=config_file_relpath, **config)
