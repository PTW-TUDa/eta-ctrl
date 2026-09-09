from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager


@pytest.fixture
def scenario_df() -> pd.DataFrame:
    start_date = pd.Timestamp("2026/01/01")
    seconds = 16
    index = pd.date_range(start_date, start_date + pd.Timedelta(seconds=seconds), freq="s")
    data = {"scen1": [2 * i for i in range(seconds + 1)]}
    return pd.DataFrame(index=index, data=data)


@pytest.fixture
def scenario_manager_factory(
    scenario_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., CsvScenarioManager]:
    def dummy_load_data(scenario_manager: CsvScenarioManager) -> None:
        scenario_manager.scenarios = scenario_df
        scenario_manager.total_df_length = len(scenario_df)

    monkeypatch.setattr(CsvScenarioManager, "load_data", dummy_load_data)

    def factory(*, use_random_time_slice: bool = False) -> CsvScenarioManager:
        return CsvScenarioManager(
            scenario_configs=[],
            start_time=scenario_df.index[0],
            end_time=scenario_df.index[-1],
            total_time=17,
            resample_time=1,
            use_random_time_slice=use_random_time_slice,
        )

    return factory


def test_initializes_attributes(
    scenario_manager_factory: Callable[..., CsvScenarioManager], scenario_df: pd.DataFrame
) -> None:
    scenario_manager = scenario_manager_factory(use_random_time_slice=True)

    assert scenario_manager.scenario_configs == []
    assert scenario_manager.start_time == scenario_df.index[0]
    assert scenario_manager.end_time == scenario_df.index[-1]
    assert scenario_manager.total_time == 17
    assert scenario_manager.resample_time == 1
    assert scenario_manager.use_random_time_slice
    assert scenario_manager.scenario_steps == 17
    assert scenario_manager.scenarios is scenario_df
    assert scenario_manager.total_df_length == len(scenario_df)


def test_random_offset_is_zero_without_available_space(
    scenario_manager_factory: Callable[..., CsvScenarioManager],
) -> None:
    scenario_manager = scenario_manager_factory(use_random_time_slice=True)

    assert scenario_manager.compute_episode_offset(np.random.default_rng(42)) == 0
