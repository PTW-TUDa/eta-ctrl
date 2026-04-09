from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from eta_ctrl.agents.mpc_agent import MpcAgent
from eta_ctrl.config.config import Config, ConfigPaths, ConfigSettings, ConfigSetup
from eta_ctrl.envs.state import StateConfig
from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager

SAMPLING_TIME = 10
EPISODE_DURATION = 60


@pytest.fixture(scope="module")
def config_params(resources_path):
    config_setup = ConfigSetup(agent_import="eta_ctrl.agents.MpcAgent", environment_import="eta_ctrl.envs.BaseEnv")
    config_settings = ConfigSettings(episode_duration=EPISODE_DURATION, sampling_time=SAMPLING_TIME)
    return {
        "root_path": resources_path,
        "config_file_relpath": "config/config2",
        "setup": config_setup,
        "settings": config_settings,
    }


class TestConfig:
    @pytest.fixture(scope="class", autouse=True)
    def config(self, class_monkeypatch, config_params, resources_path) -> Callable[..., Config]:
        # Create a path that is within root_path
        state_config_fake_path = resources_path / "test_state_config.toml"
        class_monkeypatch.setattr(
            StateConfig, "from_file", lambda **kwargs: StateConfig(source_file=state_config_fake_path)
        )
        return Config(**config_params)

    def test_default_values(self, config, resources_path):

        assert config.root_path == resources_path
        assert config.config_file_relpath == Path("config/config2")
        assert config.paths == ConfigPaths(state_file_relpath=Path("test_state_config.toml"))
        assert config.config_name == "config2"
        assert config.results_path == resources_path / "results"
        assert config.scenarios_path == resources_path / "scenarios"

    def test_all_values(self, config_params):
        """Config has one default values, config_file_relpath and paths"""
        config_extra_params = {
            "paths": ConfigPaths(state_file_relpath="config/test_env_state_config"),
            "config_file_relpath": "other/relpath",
        }
        config = Config(**{**config_params, **config_extra_params})

        assert config.paths.state_file_relpath == Path("test_state_config.toml")
        assert config.config_file_relpath == Path("other/relpath")

    def test_mpc_agent_includes_sampling_time(self, config):
        assert config.setup.agent_class is MpcAgent, "Test can't work if MpcAgent isn't chosen"
        assert "sampling_time" in config.settings.agent
        assert config.settings.agent["sampling_time"] == SAMPLING_TIME

    def test_state_config(self, config):
        assert config.settings.environment.get("state_config") is not None

    def test_str(self, config):
        str_repr = "Config 'config2' (env=BaseEnv, agent=MpcAgent)"
        assert str(config) == str_repr


class TestConfigFromFile:
    file = "config/config2"

    @pytest.fixture(scope="class", autouse=True)
    def prevent_others_loading(self, class_monkeypatch):
        class_monkeypatch.setattr(ConfigCsvScenario, "model_post_init", lambda *args: None)
        class_monkeypatch.setattr(CsvScenarioManager, "load_data", lambda *args: None)

    def test_from_file(self, resources_path):
        Config.from_file(root_path=resources_path, config_relpath="config", config_name="config2")

    def test_root_path_not_str(self, resources_path):
        Config.from_file(root_path=str(resources_path), config_relpath="config", config_name="config2")

    def test_no_root_path(self, resources_path, monkeypatch):
        monkeypatch.setattr("__main__.__file__", str(resources_path / "fake_main.py"))
        Config.from_file(config_relpath="config", config_name="config2")

    def test_from_file_overwrite(self, resources_path):
        overwrite = {"paths": {"scenarios_relpath": "data_dir"}}
        config = Config.from_file(
            root_path=resources_path, config_relpath="config", config_name="config2", overwrite=overwrite
        )
        assert config.paths.scenarios_relpath == Path("data_dir")

    def test_from_file_overwrite_nested(self, resources_path):
        overwrite = {"settings": {"agent": {"foo": "bar"}}}
        config = Config.from_file(
            root_path=resources_path, config_relpath="config", config_name="config2", overwrite=overwrite
        )
        assert config.settings.agent.get("foo") == "bar"

    def test_from_file_overwrite_error(self, resources_path):
        overwrite = {"not_existing_key": "random_value"}
        with pytest.raises(ValidationError) as excinfo:
            Config.from_file(
                root_path=resources_path, config_relpath="config", config_name="config2", overwrite=overwrite
            )
        msg = "1 validation error for Config\nnot_existing_key\n  Extra inputs are not permitted"
        assert msg in str(excinfo.value)

    def test_automatic_prediction_horizon_filling(self, resources_path, monkeypatch):
        # Mock StateConfig
        mock_file_content = {
            "observations": [{"name": "o1", "is_agent_observation": True, "duration": "n_prediction_steps"}]
        }
        monkeypatch.setattr("eta_ctrl.envs.state.load_config", lambda *args, **kwargs: mock_file_content)

        overwrite = {
            "settings": {"prediction_horizon": 60},
        }
        config = Config.from_file(
            root_path=resources_path, config_relpath="config", config_name="config2", overwrite=overwrite
        )

        assert config.settings.environment["state_config"].vars["o1"].duration == 7
