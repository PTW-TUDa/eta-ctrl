import copy
import re
from datetime import datetime
from pathlib import Path

import pytest

from eta_ctrl.config import Config, ConfigSettings, ConfigSetup
from eta_ctrl.envs.state import StateConfig
from test.resources.config.config_python import config as python_dict


@pytest.fixture
def config_dict() -> dict:
    config = copy.deepcopy(python_dict)
    config.setdefault("paths", {})["state_relpath"] = "test_env_state_config.toml"
    return config


class TestConfig:
    @pytest.fixture(autouse=True)
    def prevent_state_config_loading(self, monkeypatch):
        monkeypatch.setattr(StateConfig, "from_file", lambda path, filename: None)

    def test_from_dict(self, config_dict, config_resources_path):
        Config._from_dict(config=config_dict, config_name="test", root_path=config_resources_path)

    def test_from_dict_overwrite(self, config_dict, config_resources_path):
        overwrite = {
            "agent_specific": {"solver_name": "foobar"},
            "environment_specific": {"foo": [{"bar": "quux"}]},
        }

        config_opt = Config._from_dict(
            config=config_dict, config_name="test", root_path=config_resources_path, overwrite=overwrite
        )

        assert config_opt.settings.agent["solver_name"] == "foobar"
        assert config_opt.settings.environment["foo"][0]["bar"] == "quux"

    def test_from_dict_fail(self, config_dict, config_resources_path):
        config_dict.pop("settings")
        error_msg = re.escape("The section 'settings' is not present in configuration file foobar.")
        with pytest.raises(ValueError, match=error_msg):
            Config._from_dict(config=config_dict, config_name="foobar", root_path=config_resources_path)

    def test_build_config_dictfail(self, config_dict: dict, config_resources_path, caplog):
        caplog.set_level(10)  # Set log level to debug

        config_dict["setup"] = ["foobar"]
        config_dict.pop("environment_specific")
        error_msg = re.escape("'setup' section must be a dictionary of settings.")
        with pytest.raises(TypeError, match=error_msg):
            Config._from_dict(config_dict, config_name="", root_path=config_resources_path)
        log_msg = "Section 'environment_specific' not present in configuration, assuming it is empty."
        assert log_msg in caplog.messages

    def test_build_config_altname(self, config_dict: dict, caplog, config_resources_path):
        config_dict["interaction_environment_specific"] = {"foo": "bar"}
        config_dict.pop("interaction_env_specific", None)
        config_dict["agentspecific"] = {"solver_name": "foobar"}
        config = Config._from_dict(config_dict, config_name="", root_path=config_resources_path)
        assert config.settings.interaction_env["foo"] == "bar"
        log_msg = (
            "Specified configuration value 'agentspecific' in the setup section of the configuration "
            "was not recognized and is ignored."
        )
        assert log_msg in caplog.messages

    def test_build_config_pathfail(self, config_dict: dict, config_resources_path):
        config_dict["setup"].pop("environment_import")
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). Could not load config file."
        )
        with pytest.raises(ValueError, match=error_msg):
            Config._from_dict(config_dict, config_name="", root_path=config_resources_path)

    def test_build_config_default_path(self, config_dict: dict, config_resources_path):
        config = Config._from_dict(config_dict, config_name="", root_path=config_resources_path)
        assert config.results_path == config_resources_path / "results"
        assert config.scenarios_path == config_resources_path / "scenarios"

    def test_from_file(self, config_dict: dict, config_resources_path: Path):
        overwrite = {"paths": {"state_relpath": "test_env_state_config.toml"}}
        config = Config.from_file(
            root_path=config_resources_path, config_relpath=Path(), config_name="config1", overwrite=overwrite
        )
        assert config.config_name == "config1"


class TestConfigSetup:
    missing_classes_fail = [
        "interaction_env_class",
        "interaction_env_package",
        "vectorizer_class",
        "vectorizer_package",
        "policy_class",
        "policy_package",
    ]

    @pytest.mark.parametrize("missing_class", missing_classes_fail)
    def test_from_dict_no_fail(self, config_dict: dict, missing_class: str):
        config_dict["setup"].pop(missing_class)
        ConfigSetup.from_dict(config_dict["setup"])

    missing_classes_no_fail = [
        "environment_import",
        "agent_import",
    ]

    @pytest.mark.parametrize("missing_class", missing_classes_no_fail)
    def test_from_dict_fail(self, config_dict: dict, missing_class: str, caplog):
        config_dict["setup"].pop(missing_class)
        missing = missing_class.rsplit("_", 1)[0]
        error_msg = re.escape(
            "Not all required values were found in setup section (see log). "
            f"Could not load config file. Missing values: {missing}"
        )
        with pytest.raises(ValueError, match=error_msg):
            ConfigSetup.from_dict(config_dict["setup"])
        log_msg = (
            f"'{missing}_import' or both of '{missing}_package' and '{missing}_class' parameters must be specified."
        )
        assert log_msg in caplog.messages

    def test_module_not_found(self, config_dict: dict):
        config_dict["setup"]["environment_import"] = "foobar.FooBar"
        error_msg = "Could not find module 'foobar'. While importing class 'FooBar' from 'environment_import' value."
        with pytest.raises(ModuleNotFoundError, match=error_msg):
            ConfigSetup.from_dict(config_dict["setup"])

    def test_class_not_found(self, config_dict: dict):
        config_dict["setup"]["environment_import"] = "eta_ctrl.envs.FooBar"
        error_msg = (
            "Could not find class 'FooBar' in module 'eta_ctrl.envs'. "
            "While importing class 'FooBar' from 'environment_import' value."
        )
        with pytest.raises(AttributeError, match=error_msg):
            ConfigSetup.from_dict(config_dict["setup"])

    def test_unrecognized_keys(self, config_dict: dict, caplog):
        config_dict["setup"]["foobar"] = "barfoo"
        ConfigSetup.from_dict(config_dict["setup"])
        log_msg = "Following values were not recognized in the config setup section and are ignored: foobar"
        assert log_msg in caplog.messages


class TestConfigSettings:
    @pytest.mark.parametrize("missing_key", ["n_episodes_play", "episode_duration", "sampling_time"])
    def test_from_dict_fail(self, config_dict: dict, missing_key: str):
        config_dict["settings"].pop(missing_key)
        error_msg = re.escape("Not all required values were found in settings (see log). Could not load config file.")
        with pytest.raises(ValueError, match=error_msg):
            ConfigSettings.from_dict(config_dict)

    def test_scenario_time_begin_end_with_seconds(self, config_dict: dict):
        config_dict["settings"]["scenario_time_begin"] = "2022-03-18 00:00:00"
        config_dict["settings"]["scenario_time_end"] = "2022-03-18 00:10:00"
        settings = ConfigSettings.from_dict(config_dict)
        assert settings.scenario_time_begin == datetime(2022, 3, 18, 0, 0, 0)
        assert settings.scenario_time_end == datetime(2022, 3, 18, 0, 10, 0)

    def test_scenario_time_begin_end_without_seconds(self, config_dict: dict):
        config_dict["settings"]["scenario_time_begin"] = "2022-03-18 00:00"
        config_dict["settings"]["scenario_time_end"] = "2022-03-18 00:10"
        settings = ConfigSettings.from_dict(config_dict)
        assert settings.scenario_time_begin == datetime(2022, 3, 18, 0, 0, 0)
        assert settings.scenario_time_end == datetime(2022, 3, 18, 0, 10, 0)
