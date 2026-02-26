from __future__ import annotations

import pathlib
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import define, field, validators

import __main__
from eta_ctrl.envs.state import StateConfig
from eta_ctrl.util import deep_mapping_update
from eta_ctrl.util.io_utils import load_config
from eta_ctrl.util.utils import camel_to_snake_case

from .config_settings import ConfigSettings
from .config_setup import ConfigSetup

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from eta_ctrl.util.type_annotations import Path


# Helper extracted to reduce branching in _from_dict
def _pop_dict(dikt: dict, key: str) -> dict:
    val = dikt.pop(key)
    if not isinstance(val, dict):
        # Prefer TypeError for invalid types
        msg = f"'{key}' section must be a dictionary of settings."
        raise TypeError(msg)
    return val


def _derive_state_config(root_path: pathlib.Path, paths: dict, setup: ConfigSetup) -> StateConfig:
    state_file = camel_to_snake_case(setup.environment_class.__name__) + "_state_config"
    state_path = root_path / paths.pop("state_relpath", "")
    return StateConfig.from_file(path=state_path, filename=state_file)


def _path_or_default(value: str | pathlib.Path | None, default: str) -> pathlib.Path:
    """Convert a possibly-None value into a pathlib.Path using a default.

    This helper ensures attrs converters never receive ``None`` which would make
    ``pathlib.Path(None)`` raise a TypeError.
    """
    if value is None:
        return pathlib.Path(default)
    return pathlib.Path(value)


def _convert_results_relpath(value: str | pathlib.Path | None) -> pathlib.Path:
    return _path_or_default(value, "results")


def _convert_scenarios_relpath(value: str | pathlib.Path | None) -> pathlib.Path:
    return _path_or_default(value, "scenarios")


log = getLogger(__name__)


@define(frozen=False, kw_only=True)
class Config:
    """Configuration for the optimization, which can be loaded from a JSON file."""

    #: Name of the configuration used for the series of run.
    config_name: str = field(validator=validators.instance_of(str))

    #: Root folder path for the optimization run (default: parent folder of invoking script).
    root_path: pathlib.Path = field(converter=pathlib.Path)
    #: Relative path to the results folder (default: results).
    results_relpath: pathlib.Path = field(converter=_convert_results_relpath, default=pathlib.Path("results"))
    #: relative path to the scenarios folder (default: scenarios).
    scenarios_relpath: pathlib.Path = field(converter=_convert_scenarios_relpath, default=pathlib.Path("scenarios"))
    #: Path to the results folder (default: root_path/results).
    results_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the scenarios folder (default: root_path/scenarios).
    scenarios_path: pathlib.Path = field(init=False, converter=pathlib.Path)

    #: Optimization run setup.
    setup: ConfigSetup = field()
    #: Optimization run settings.
    settings: ConfigSettings = field()

    def __attrs_post_init__(self) -> None:
        if not self.root_path.exists():
            msg = f"Root path {self.root_path} in config {self.config_name} does not exist"
            raise ValueError(msg)

        # compute and assign resolved paths
        self.results_path = self.root_path / self.results_relpath
        self.scenarios_path = self.root_path / self.scenarios_relpath

        self.settings.create_scenario_manager(self.scenarios_path)

    @classmethod
    def from_file(
        cls,
        root_path: Path | None,
        config_relpath: Path | None,
        config_name: str,
        overwrite: Mapping[str, Any] | None = None,
    ) -> Config:
        """Load configuration from JSON/TOML/YAML file, which consists of the following sections:

        - **paths**: In this section, the (relative) file paths for results and scenarios are specified. The paths
          are deserialized directly into the :class:`Config` object.
        - **setup**: This section specifies which classes and utilities should be used for optimization. The setup
          configuration is deserialized into the :class:`ConfigSetup` object.
        - **settings**: The settings section contains basic parameters for the optimization, it is deserialized
          into a :class:`ConfigSettings` object.
        - **environment_specific**: The environment section contains keyword arguments for the environment.
          This section must contain values for the arguments of the environment, the expected values are therefore
          different depending on the environment and not fully documented here.
        - **agent_specific**: The agent section contains keyword arguments for the control algorithm (agent).
          This section must contain values for the arguments of the agent, the expected values are therefore
          different depending on the agent and not fully documented here.

        :param root_path: Path to the experiment root.
        :param config_relpath: Path to the configuration directory, relative to root_path.
        :param config_name: Name of the configuration file, without extension.
        :param overwrite: Config parameters to overwrite.
        :return: Config object.
        """
        if root_path is None:
            # Use parent folder of invoking script when root_path is not provided
            root_path = pathlib.Path(__main__.__file__).parent.resolve()
        elif not isinstance(root_path, pathlib.Path):
            root_path = pathlib.Path(root_path)

        if config_relpath is None:
            config_relpath = pathlib.Path("config")

        if not root_path.exists():
            msg = f"Root path {root_path} does not exist"
            raise ValueError(msg)

        file_path = root_path / config_relpath / f"{config_name}"
        config = load_config(file_path)

        return Config._from_dict(config=config, root_path=root_path, config_name=config_name, overwrite=overwrite)

    @classmethod
    def _from_dict(
        cls,
        config: dict[str, Any],
        config_name: str,
        root_path: pathlib.Path,
        overwrite: Mapping[str, Any] | None = None,
    ) -> Config:
        """Build a Config object from a dictionary of configuration options.

        :param config: Dictionary of configuration options.
        :param file: Path to the configuration file.
        :param root_path: Root path for the optimization configuration run.
        :return: Config object.
        """

        if overwrite is not None:
            config = dict(deep_mapping_update(config, overwrite))

        # Ensure required sections are present
        for section in ("setup", "settings"):
            if section not in config:
                msg = f"The section '{section}' is not present in configuration file {config_name}."
                raise ValueError(msg)

        # Provide empty dicts for optional sections if missing
        for section in ("environment_specific", "agent_specific", "paths"):
            if section not in config:
                config[section] = {}
                log.info(f"Section '{section}' not present in configuration, assuming it is empty.")

        # Load paths section
        paths = _pop_dict(config, "paths")
        results_relpath = paths.pop("results_relpath", None)
        scenarios_relpath = paths.pop("scenarios_relpath", None)

        # Load settings section
        settings_raw: dict[str, dict[str, Any]] = {}
        settings_raw["settings"] = _pop_dict(config, "settings")
        settings_raw["environment_specific"] = _pop_dict(config, "environment_specific")

        # Create ConfigSetup
        _setup = _pop_dict(config, "setup")
        setup = ConfigSetup.from_dict(_setup)

        # Create StateConfig (moved to helper to lower function complexity)
        state_config = _derive_state_config(root_path, paths, setup)
        settings_raw["environment_specific"]["state_config"] = state_config

        if "interaction_env_specific" in config:
            settings_raw["interaction_env_specific"] = _pop_dict(config, "interaction_env_specific")
        elif "interaction_environment_specific" in config:
            settings_raw["interaction_env_specific"] = _pop_dict(config, "interaction_environment_specific")

        settings_raw["agent_specific"] = _pop_dict(config, "agent_specific")

        # Log unrecognized values
        for name in config:
            log.warning(
                f"Specified configuration value '{name}' in the setup section of the configuration was not "
                f"recognized and is ignored."
            )

        return cls(
            config_name=config_name,
            root_path=root_path,
            results_relpath=results_relpath,
            scenarios_relpath=scenarios_relpath,
            setup=setup,
            settings=ConfigSettings.from_dict(settings_raw),
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            msg = f"The key {name} does not exist - it cannot be set."
            raise KeyError(msg)
        setattr(self, name, value)
