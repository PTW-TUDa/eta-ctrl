"""Tests for ConfigRun."""

import pathlib
import tempfile

import pytest

from eta_ctrl.config import ConfigRun


class TestConfigRunStringRepresentations:
    """Tests for ConfigRun.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def config_run(self):
        temp_path = pathlib.Path(tempfile.mkdtemp())
        return ConfigRun(
            series="my_series",
            name="my_run",
            description="test description",
            root_path=temp_path,
            results_path=temp_path / "results",
            scenarios_path=temp_path / "scenarios",
        )

    # --- __str__ ---

    def test_str_exact(self, config_run):
        assert str(config_run) == "ConfigRun(series='my_series', name='my_run')"

    def test_str_contains_series_and_name(self, config_run):
        result = str(config_run)
        assert "series='my_series'" in result
        assert "name='my_run'" in result
