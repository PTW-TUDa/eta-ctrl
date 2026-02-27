import numpy as np
import pandas as pd
import pytest

from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager
from eta_ctrl.timeseries.scenarios import _fix_col_name

FIX_COL_NAME_TEST_CASES = [
    # name, prefix, prefix_renamed, rename_cols, expected_result, test_id
    # Basic cases, no renaming
    ("col", None, False, None, "col", "no_prefix_no_rename"),
    ("col", "pre", False, None, "pre_col", "prefix_no_rename"),
    # With renaming, prefix_renamed=False
    ("old", None, False, {"old": "new"}, "new", "rename_no_prefix_at_all"),
    ("old", "pre", False, {"old": "new"}, "new", "rename_no_prefix_on_renamed"),
    # With renaming, prefix_renamed=True
    ("old", None, True, {"old": "new"}, "new", "rename_prefix_renamed_but_no_prefix"),
    ("old", "pre", True, {"old": "new"}, "pre_new", "rename_with_prefix_on_renamed"),
    # Edge cases
    ("", "pre", False, None, "pre_", "empty_name_with_prefix"),
    ("col", "", False, None, "_col", "empty_prefix"),
    ("col", "pre", False, {}, "pre_col", "empty_rename_dict"),
]


@pytest.mark.parametrize(
    ("name", "prefix", "prefix_renamed", "rename_cols", "expected", "test_id"),
    FIX_COL_NAME_TEST_CASES,
    ids=[case[5] for case in FIX_COL_NAME_TEST_CASES],  # Use id for readable output
)
def test_fix_col_name(name, prefix, prefix_renamed, rename_cols, expected, test_id):
    """Test _fix_col_name with various parameter combinations."""
    result = _fix_col_name(name=name, prefix=prefix, prefix_renamed=prefix_renamed, rename_cols=rename_cols)
    assert result == expected


@pytest.fixture
def scenario_df():
    start_date = pd.Timestamp("2026/01/01")
    seconds = 16
    index = pd.date_range(start_date, start_date + pd.Timedelta(seconds=seconds), freq="s")
    data = {"scen1": [2 * i for i in range(seconds + 1)]}  # 0, 2, 4, ..., 2*i
    return pd.DataFrame(index=index, data=data)


@pytest.fixture
def scenario_manager_factory(scenario_df, monkeypatch):
    def factory(use_random_time_slice: bool):
        def dummy_load_data(scenario_manager):
            scenario_manager.scenarios = scenario_df
            scenario_manager.total_df_length = len(scenario_df)

        monkeypatch.setattr(CsvScenarioManager, "load_data", dummy_load_data)
        return CsvScenarioManager(
            scenario_configs=[],
            start_time=scenario_df.index[0],
            end_time=scenario_df.index[-1],
            total_time=17,
            resample_time=1,
            use_random_time_slice=use_random_time_slice,
        )

    return factory


def test_scenario_manager_random_no_space(scenario_manager_factory):
    sm = scenario_manager_factory(use_random_time_slice=True)
    rng = np.random.default_rng(42)
    assert sm.compute_episode_offset(rng) == 0
