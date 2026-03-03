"""Tests for __str__ and __repr__ of CsvScenarioManager."""

import pandas as pd
import pytest

from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager

# Shared time constants
_START_TIME = pd.Timestamp("2022-03-18 00:00")
_END_TIME = _START_TIME + pd.Timedelta(hours=2)
_TOTAL_TIME = pd.Timedelta(hours=2).total_seconds()
_RESAMPLE_TIME = pd.Timedelta(minutes=10).total_seconds()


def _make_scenario_config(path: str, scenarios_path) -> ConfigCsvScenario:
    """Helper to create a ConfigCsvScenario from a relative path."""
    return ConfigCsvScenario(
        path=path,
        interpolation_method="ffill",
        scenarios_path=scenarios_path,
    )


def _make_manager(scenario_configs: list, **kwargs) -> CsvScenarioManager:
    """Helper to create a CsvScenarioManager with default time settings."""
    return CsvScenarioManager(
        scenario_configs=scenario_configs,
        start_time=_START_TIME,
        end_time=_END_TIME,
        total_time=_TOTAL_TIME,
        resample_time=_RESAMPLE_TIME,
        **kwargs,
    )


class TestCsvScenarioManagerStringRepresentations:
    """Tests for CsvScenarioManager.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def scenarios_path(self, config_resources_path):
        return config_resources_path / "scenarios"

    @pytest.fixture(scope="class")
    def single_scenario_manager(self, scenarios_path):
        cfg = _make_scenario_config("electricity_price_test.csv", scenarios_path)
        return _make_manager([cfg])

    @pytest.fixture(scope="class")
    def multi_scenario_manager(self, scenarios_path):
        configs = [
            _make_scenario_config("electricity_price_test.csv", scenarios_path),
            _make_scenario_config("extra_dir/test_data.csv", scenarios_path),
        ]
        return _make_manager(configs)

    # --- __str__ ---

    def test_str_single_scenario_exact(self, single_scenario_manager):
        expected = "CsvScenarioManager(1 scenario(s), 2022-03-18 00:00:00 to 2022-03-18 02:00:00)"
        assert str(single_scenario_manager) == expected

    def test_str_multi_scenario_exact(self, multi_scenario_manager):
        expected = "CsvScenarioManager(2 scenario(s), 2022-03-18 00:00:00 to 2022-03-18 02:00:00)"
        assert str(multi_scenario_manager) == expected

    def test_str_contains_scenario_count(self, single_scenario_manager, multi_scenario_manager):
        assert "1 scenario(s)" in str(single_scenario_manager)
        assert "2 scenario(s)" in str(multi_scenario_manager)

    def test_str_contains_time_range(self, single_scenario_manager):
        result = str(single_scenario_manager)
        assert "2022-03-18 00:00:00" in result
        assert "2022-03-18 02:00:00" in result

    # --- __repr__ ---

    def test_repr_single_scenario_exact(self, single_scenario_manager):
        expected = (
            f"CsvScenarioManager(start_time={_START_TIME!r}, end_time={_END_TIME!r}, "
            f"total_time={_TOTAL_TIME}, n_scenarios=1, "
            f"columns={list(single_scenario_manager.scenarios.columns)})"
        )
        assert repr(single_scenario_manager) == expected

    def test_repr_contains_columns(self, single_scenario_manager):
        """Verify columns field is present and reflects the loaded scenario data."""
        result = repr(single_scenario_manager)
        assert f"columns={list(single_scenario_manager.scenarios.columns)}" in result

    def test_repr_multi_scenario_shows_correct_count(self, multi_scenario_manager):
        assert "n_scenarios=2" in repr(multi_scenario_manager)

    def test_repr_contains_total_time(self, single_scenario_manager):
        assert f"total_time={_TOTAL_TIME}" in repr(single_scenario_manager)

    def test_repr_contains_start_and_end_time(self, single_scenario_manager):
        result = repr(single_scenario_manager)
        assert f"start_time={_START_TIME!r}" in result
        assert f"end_time={_END_TIME!r}" in result
