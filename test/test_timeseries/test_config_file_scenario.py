import copy
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from eta_ctrl.config.config import Config
from eta_ctrl.envs.state import StateConfig
from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager
from test.resources.config.config_python import config as python_dict


@pytest.fixture(scope="module")
def raw_scenarios():
    return [
        {
            "path": "electricity_price_test.csv",
            "interpolation_method": "ffill",
            "time_conversion_str": "%d.%m.%Y %H:%M",
            "prefix": "ep",
            "scale_factors": {"Electrical_Energy_Price_MWh": 0.001},
        },
        {
            "path": "extra_dir/test_data.csv",
            "infer_datetime_cols": "dates",
            "prefix": "test",
            "scale_factors": {"foo": 1, "quux": 2},
            "rename_cols": {"quux": "xuuq"},
        },
    ]


def test_multiple(raw_scenarios, config_resources_path):
    path_scenarios = config_resources_path / "scenarios"
    scenario_configs = [
        ConfigCsvScenario(**raw_scenario, path_scenarios=path_scenarios) for raw_scenario in raw_scenarios
    ]
    start_time = pd.Timestamp("2022/03/18 0:00")
    data_manager = CsvScenarioManager(
        scenario_configs=scenario_configs,
        start_time=start_time,
        end_time=start_time + pd.Timedelta(hours=2),
        resample_time=pd.Timedelta(minutes=10).total_seconds(),
        total_time=pd.Timedelta(hours=2).total_seconds(),  # 2h funktioniert nicht
    )
    data = data_manager.scenarios

    assert set(data.columns) == {"ep_Electrical_Energy_Price_MWh", "test_foo_barbaz", "test_xuuq"}

    assert (data["ep_Electrical_Energy_Price_MWh"] == [1, 1, 1, 1, 9, 9, 9, 2, 2, 6, 6, 1, 1]).all()
    assert (data["test_foo_barbaz"] == [10 + 20 * i for i in range(13)]).all()
    assert (data["test_xuuq"] == [198 for _ in range(13)]).all()


@pytest.mark.parametrize(
    "argument",
    [
        {"interpolation_method": "foo"},
        {"scale_factors": [2]},
        {"infer_datetime_cols": "foo"},
        {"unknown_param": "foo"},
    ],
    ids=lambda param: f"arg_{next(iter(param.keys()))}",
)
def test_wrong_arguments(monkeypatch, argument):
    monkeypatch.setattr(ConfigCsvScenario, "model_post_init", lambda *_: ...)

    default_args = {"path": "", "path_scenarios": Path()}
    with pytest.raises(ValidationError):
        ConfigCsvScenario(**default_args, **argument)


def test_path_not_found():
    with pytest.raises(FileNotFoundError):
        ConfigCsvScenario(path="foo", path_scenarios=Path())


#############


@pytest.fixture(autouse=True)
def prevent_state_config_loading(monkeypatch):
    monkeypatch.setattr(StateConfig, "from_file", lambda _: None)


@pytest.fixture
def config(config_resources_path):
    return Config.from_dict(config=copy.deepcopy(python_dict), config_name="test", path_root=config_resources_path)


def test_init_from_config(config: Config):
    scenario_manager = config.settings.environment["scenario_manager"]
    config_scenario = scenario_manager.scenario_configs
    electricity_price = config_scenario[0]
    assert electricity_price.abs_path == config.path_scenarios / electricity_price.path
    assert electricity_price.interpolation_method == "ffill"
    assert electricity_price.time_conversion_str == "%d.%m.%Y %H:%M"
