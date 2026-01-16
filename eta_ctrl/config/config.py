from __future__ import annotations

import pathlib
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import converters, define, field, validators

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


log = getLogger(__name__)


@define(frozen=False, kw_only=True)
class Config:
    """Configuration for the optimization, which can be loaded from a JSON file."""

    #: Name of the configuration used for the series of run.
    config_name: str = field(validator=validators.instance_of(str))

    #: Root path for the optimization run (scenarios and results are relative to this).
    path_root: pathlib.Path = field(converter=pathlib.Path)
    #: Relative path to the state config.
    relpath_state: pathlib.Path = field(converter=pathlib.Path)
    #: Relative path to the results folder.
    relpath_results: pathlib.Path = field(converter=pathlib.Path)
    #: relative path to the scenarios folder (default: None).
    relpath_scenarios: pathlib.Path | None = field(converter=converters.optional(pathlib.Path), default=None)
    #: Path to the results folder.
    path_results: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the scenarios folder (default: None).
    path_scenarios: pathlib.Path | None = field(init=False, converter=converters.optional(pathlib.Path), default=None)

    #: Optimization run setup.
    setup: ConfigSetup = field()
    #: Optimization run settings.
    settings: ConfigSettings = field()

    def __attrs_post_init__(self) -> None:
        if not self.path_root.exists():
            msg = f"Root path {self.path_root} in config {self.config_name} does not exist"
            raise ValueError(msg)
        object.__setattr__(self, "path_results", self.path_root / self.relpath_results)

        if self.relpath_scenarios is not None:
            self.path_scenarios = self.path_root / self.relpath_scenarios

        self.settings.create_scenario_manager(self.path_scenarios)

    @classmethod
    def from_file(cls, file: Path, path_root: Path, overwrite: Mapping[str, Any] | None = None) -> Config:
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

        :param file: Path to the configuration file.
        :param overwrite: Config parameters to overwrite.
        :return: Config object.
        """
        config = load_config(file)
        config_name = pathlib.Path(file).stem

        return Config.from_dict(config, config_name, path_root, overwrite)

    @classmethod
    def from_dict(
        cls, config: dict[str, Any], config_name: str, path_root: Path, overwrite: Mapping[str, Any] | None = None
    ) -> Config:
        """Build a Config object from a dictionary of configuration options.

        :param config: Dictionary of configuration options.
        :param file: Path to the configuration file.
        :param path_root: Root path for the optimization configuration run.
        :return: Config object.
        """
        if overwrite is not None:
            config = dict(deep_mapping_update(config, overwrite))

        # Ensure all required sections are present in configuration
        for section in ("setup", "settings", "paths"):
            if section not in config:
                msg = f"The section '{section}' is not present in configuration file {config_name}."
                raise ValueError(msg)

        def _pop_dict(dikt: dict[str, Any], key: str) -> dict[str, Any]:
            val = dikt.pop(key)
            if not isinstance(val, dict):
                msg = f"'{key}' section must be a dictionary of settings."
                raise TypeError(msg)
            return val

        if "environment_specific" not in config:
            config["environment_specific"] = {}
            log.info("Section 'environment_specific' not present in configuration, assuming it is empty.")

        if "agent_specific" not in config:
            config["agent_specific"] = {}
            log.info("Section 'agent_specific' not present in configuration, assuming it is empty.")

        # Load values from paths section
        errors = False
        paths = _pop_dict(config, "paths")

        if "relpath_results" not in paths:
            log.error("'relpath_results' is required and could not be found in section 'paths' of the configuration.")
            errors = True
        relpath_results = paths.pop("relpath_results", None)
        relpath_scenarios = paths.pop("relpath_scenarios", None)

        # Setup section
        _setup = _pop_dict(config, "setup")
        try:
            setup = ConfigSetup.from_dict(_setup)
        except ValueError:
            log.exception("Failed creating ConfigSetup object")
            errors = True

        # Settings section
        settings_raw: dict[str, dict[str, Any]] = {}
        settings_raw["settings"] = _pop_dict(config, "settings")
        settings_raw["environment_specific"] = _pop_dict(config, "environment_specific")

        relpath_state = paths.pop("relpath_state", None)
        if relpath_state is None:
            # Retrieve environment name and look for associated StateConfig file
            snake_case_name = camel_to_snake_case(setup.environment_class.__name__)
            relpath_state = f"{snake_case_name}_state_config"
            log.info(f"Loading StateConfig from default path at {relpath_state}.")
        path_state = path_root / relpath_state

        state_config = StateConfig.from_file(path_state)
        settings_raw["environment_specific"]["state_config"] = state_config

        if "interaction_env_specific" in config:
            settings_raw["interaction_env_specific"] = _pop_dict(config, "interaction_env_specific")
        elif "interaction_environment_specific" in config:
            settings_raw["interaction_env_specific"] = _pop_dict(config, "interaction_environment_specific")

        settings_raw["agent_specific"] = _pop_dict(config, "agent_specific")

        try:
            settings = ConfigSettings.from_dict(settings_raw)
        except ValueError:
            log.exception("Failed creating ConfigSettings object")
            errors = True
        # Log configuration values which were not recognized.
        for name in config:
            log.warning(
                f"Specified configuration value '{name}' in the setup section of the configuration was not "
                f"recognized and is ignored."
            )

        if errors:
            msg = "Not all required values were found in setup section (see log). Could not load config file."
            raise ValueError(msg)

        return cls(
            config_name=config_name,
            path_root=path_root,
            relpath_results=relpath_results,
            relpath_scenarios=relpath_scenarios,
            relpath_state=relpath_state,
            setup=setup,
            settings=settings,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            msg = f"The key {name} does not exist - it cannot be set."
            raise KeyError(msg)
        setattr(self, name, value)
