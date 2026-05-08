import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from eta_ctrl.config.config import ConfigSettings
from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager

scenario_time_begin = datetime(year=2026, month=1, day=1)
scenario_time_end = scenario_time_begin + timedelta(hours=1)
config_csv_scenario = [ConfigCsvScenario(path="file1.csv")]
SAMPLING_TIME = 10
EPISODE_DURATION = 600
PREDICTION_HORIZON = 60
config = {
    "settings": {
        "seed": 42,
        "verbosity": 1,
        "n_environments": 2,
        "n_episodes_play": 3,
        "n_episodes_learn": 4,
        "save_model_every_x_episodes": 5,
        "plot_interval": 5,
        "scenario_time_begin": scenario_time_begin,
        "scenario_time_end": scenario_time_end,
        "use_random_time_slice": True,
        "sampling_time": SAMPLING_TIME,
        "episode_duration": EPISODE_DURATION,
        "prediction_horizon": PREDICTION_HORIZON,
        "sim_steps_per_sample": 1,
        "scale_actions": 1,
        "round_actions": 1,
        "log_to_file": False,
        "scenario_files": config_csv_scenario,
    },
}


class TestConfigSettings:
    def test_default_values(self):
        wanted_values = {"sampling_time", "episode_duration"}
        config_settings = ConfigSettings(**{k: config["settings"][k] for k in config["settings"] if k in wanted_values})

        assert config_settings.seed is None
        assert config_settings.verbose == 2
        assert config_settings.n_environments == 1
        assert config_settings.n_episodes_play == 1
        assert config_settings.n_episodes_learn == 1
        assert config_settings.save_model_every_x_episodes == 10
        assert config_settings.plot_interval == 10
        assert config_settings.scenario_time_begin is None
        assert config_settings.scenario_time_end is None
        assert not config_settings.use_random_time_slice
        assert config_settings.sampling_time == SAMPLING_TIME
        assert config_settings.episode_duration == EPISODE_DURATION
        assert config_settings.prediction_horizon is None
        assert config_settings.scale_actions is None
        assert config_settings.round_actions is None
        assert config_settings.log_to_file
        assert config_settings.scenario_files is None

        assert config_settings.environment["verbose"] == 2
        assert config_settings.environment["sampling_time"] == SAMPLING_TIME
        assert config_settings.environment["episode_duration"] == EPISODE_DURATION

        assert config_settings.agent["seed"] is None
        assert config_settings.agent["verbose"] == 2

    def test_all_values(self):
        config_settings = ConfigSettings(**config["settings"])

        assert config_settings.seed == 42
        assert config_settings.verbose == 1
        assert config_settings.n_environments == 2
        assert config_settings.n_episodes_play == 3
        assert config_settings.n_episodes_learn == 4
        assert config_settings.save_model_every_x_episodes == 5
        assert config_settings.plot_interval == 5
        assert config_settings.scenario_time_begin == scenario_time_begin
        assert config_settings.scenario_time_end == scenario_time_begin + timedelta(hours=1)
        assert config_settings.use_random_time_slice
        assert config_settings.sampling_time == SAMPLING_TIME
        assert config_settings.episode_duration == EPISODE_DURATION
        assert config_settings.prediction_horizon == PREDICTION_HORIZON
        assert config_settings.scale_actions == 1
        assert config_settings.round_actions == 1
        assert not config_settings.log_to_file
        assert config_settings.scenario_files == config_csv_scenario
        assert config_settings.n_prediction_steps == int(PREDICTION_HORIZON / SAMPLING_TIME) + 1

        assert config_settings.environment["verbose"] == 1
        assert config_settings.environment["sampling_time"] == SAMPLING_TIME
        assert config_settings.environment["episode_duration"] == EPISODE_DURATION

        assert config_settings.agent["seed"] == 42
        assert config_settings.agent["verbose"] == 1

    def test_alias_values(self):
        extra_params = {"env_specific": {"foo": "bar"}, "agent_specific": {"verbose": "3"}}
        config_settings = ConfigSettings(**config["settings"], **extra_params)
        assert config_settings.environment["foo"] == "bar"
        assert config_settings.agent["verbose"] == "3"

    def test_extra_values(self, caplog):
        ConfigSettings(**config["settings"], extra_parameter="foo")
        msg = "Following values were not recognized in the config settings section and are ignored: extra_parameter"
        assert msg in caplog.messages

    def test_str(self):
        config_settings = ConfigSettings(**config["settings"])
        str_repr = "ConfigSettings(episode_duration=600.0, sampling_time=10.0, n_environments=2)"
        assert str(config_settings) == str_repr


class TestConfigSettingsScenarioManager:
    """Test whether the correct ScenarioManager is produced."""

    @pytest.fixture(scope="class")
    def config_settings_factory(self, class_monkeypatch) -> Callable[..., ConfigSettings]:
        class_monkeypatch.setattr(ConfigCsvScenario, "model_post_init", lambda *args: None)
        class_monkeypatch.setattr(CsvScenarioManager, "load_data", lambda *args: None)

        def factory(**extra_params):
            return ConfigSettings(**{**config["settings"], **extra_params})

        return factory

    @pytest.fixture(scope="class")
    def scenario_manager_factory(self, config_settings_factory) -> Callable[..., CsvScenarioManager]:
        def factory(**extra_params):
            config_settings = config_settings_factory(**extra_params)
            config_settings.create_scenario_manager(scenarios_path=Path())
            return config_settings.environment["scenario_manager"]

        return factory

    def test_attributes(self, scenario_manager_factory: Callable[..., CsvScenarioManager]):
        scenario_manager = scenario_manager_factory()

        assert scenario_manager.start_time == scenario_time_begin
        assert scenario_manager.end_time == scenario_time_end
        assert scenario_manager.resample_time == SAMPLING_TIME
        assert scenario_manager.use_random_time_slice

    def test_with_prediction_horizon(self, scenario_manager_factory: Callable[..., CsvScenarioManager]):
        scenario_manager = scenario_manager_factory()
        assert scenario_manager.total_time == EPISODE_DURATION + PREDICTION_HORIZON

    def test_without_prediction_horizon(self, scenario_manager_factory: Callable[..., CsvScenarioManager]):
        scenario_manager = scenario_manager_factory(prediction_horizon=None)
        assert scenario_manager.total_time == EPISODE_DURATION + SAMPLING_TIME

    # Fail cases / early return
    def test_no_scenario_files(self, config_settings_factory: Callable[..., ConfigSettings]):
        config_settings = config_settings_factory(scenario_files=None)
        config_settings.create_scenario_manager(Path())
        assert "scenario_manager" not in config_settings.environment

    def test_missing_date(self, config_settings_factory: Callable[..., ConfigSettings]):
        config_settings = config_settings_factory(scenario_time_end=None)
        msg = "Define scenario_time_begin and scenario_time_end in config [settings] section when using scenarios."
        with pytest.raises(TypeError, match=re.escape(msg)):
            config_settings.create_scenario_manager(Path())

    def test_end_date_before(self, config_settings_factory: Callable[..., ConfigSettings]):
        config_settings = config_settings_factory(scenario_time_end=scenario_time_begin - timedelta(hours=1))
        msg = "scenario_time_begin must be smaller than or equal to scenario_time_end."
        with pytest.raises(ValueError, match=msg):
            config_settings.create_scenario_manager(Path())

    def test_duration_too_short(self, config_settings_factory: Callable[..., ConfigSettings]):
        config_settings = config_settings_factory(scenario_time_end=scenario_time_begin + timedelta(minutes=1))
        msg = (
            "Given scenario time range from 2026-01-01 00:00:00 to 2026-01-01 00:01:00 "
            "does not cover the requested duration of 660.0 seconds."
        )
        with pytest.raises(ValueError, match=msg):
            config_settings.create_scenario_manager(Path())


class TestConfigSettingsFail:
    def test_not_divisible_sampling_time(self, caplog):
        params = {**config["settings"], "episode_duration": 15}
        config_setup = ConfigSettings(**params)
        msg = "Episode duration 15.0 is not a multiple of sampling time 10.0. Rounding down to 10.0."
        assert msg in caplog.messages
        assert config_setup.episode_duration == 10

    def test_convert_datetime(self):
        date_str = "2026-01-01T00:00"
        params = {**config["settings"], "scenario_time_begin": date_str}
        config_setup = ConfigSettings(**params)
        assert config_setup.scenario_time_begin == datetime.fromisoformat(date_str)

    def test_double_alias_value(self):
        with pytest.raises(ValidationError) as excinfo:
            ConfigSettings(**config["settings"], env={"foo": 2}, environment={"foo": 1})
        msg = "Multiple keys for 'environment' settings found:"
        assert msg in str(excinfo.value)
        msg = "Use only 'environment'."
        assert msg in str(excinfo.value)

    @pytest.mark.parametrize("missing_key", ["episode_duration", "sampling_time"])
    def test_from_dict_fail(self, missing_key: str):
        fail_config = {k: v for k, v in config["settings"].items() if k != missing_key}

        with pytest.raises(ValidationError) as excinfo:
            ConfigSettings(**fail_config)
        msg = f"1 validation error for ConfigSettings\n{missing_key}\n  Field required"
        assert msg in str(excinfo.value)
