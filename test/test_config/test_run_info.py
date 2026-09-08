"""Tests for RunInfo."""

import pathlib
import tempfile

import pytest

from eta_ctrl.config import ConfigPaths, RunInfo


class TestRunInfoStringRepresentations:
    """Tests for RunInfo.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def run_info(self):
        temp_path = pathlib.Path(tempfile.mkdtemp())
        return RunInfo(
            series="my_series",
            name="my_run",
            description="test description",
            root_path=temp_path,
            paths=ConfigPaths(),
        )

    # --- __str__ ---

    def test_str_exact(self, run_info):
        assert str(run_info) == "RunInfo(series='my_series', name='my_run')"

    def test_str_contains_series_and_name(self, run_info):
        result = str(run_info)
        assert "series='my_series'" in result
        assert "name='my_run'" in result

    def test_absolute_paths(self, run_info: RunInfo):
        assert run_info.results_path == run_info.root_path / "results"
        assert run_info.scenarios_path == run_info.root_path / "scenarios"
        assert run_info.series_results_path == run_info.results_path / "my_series"
        assert run_info.run_model_path == run_info.series_results_path / "my_run_model.zip"
        assert run_info.run_info_path == run_info.series_results_path / "my_run_info.json"
        assert run_info.vec_normalize_path == run_info.series_results_path / "vec_normalize.pkl"
        assert run_info.net_arch_path == run_info.series_results_path / "net_arch.txt"
        assert run_info.log_output_path == run_info.series_results_path / "my_run_log_output.log"
        assert run_info.models_path == run_info.series_results_path / "models"

    def test_run_monitor_path_deprecated(self, run_info: RunInfo):
        with pytest.warns(DeprecationWarning, match="run_monitor_path"):
            assert run_info.run_monitor_path == run_info.series_results_path / "my_run_monitor.csv"

    def test_run_monitor_path_not_in_model_dump(self, run_info: RunInfo):
        assert "run_monitor_path" not in run_info.model_dump()


@pytest.mark.parametrize(
    ("attribute_name", "invalid_name"),
    [
        ("series", "my/series"),
        ("series", "my\\series"),
        ("name", "my/run"),
        ("name", "my\\run"),
    ],
)
def test_run_info_rejects_path_separators(attribute_name, invalid_name, tmp_path):
    run_info_kwargs = {
        "series": "my_series",
        "name": "my_run",
        "description": "test description",
        "root_path": tmp_path,
        "paths": ConfigPaths(),
    }
    run_info_kwargs[attribute_name] = invalid_name

    with pytest.raises(ValueError, match="must not contain path separators"):
        RunInfo(**run_info_kwargs)
