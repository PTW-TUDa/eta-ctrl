"""Tests for RunInfo."""

import pathlib
import tempfile

import pytest

from eta_ctrl.config import RunInfo


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
            results_path=temp_path / "results",
            scenarios_path=temp_path / "scenarios",
        )

    # --- __str__ ---

    def test_str_exact(self, run_info):
        assert str(run_info) == "RunInfo(series='my_series', name='my_run')"

    def test_str_contains_series_and_name(self, run_info):
        result = str(run_info)
        assert "series='my_series'" in result
        assert "name='my_run'" in result


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
        "results_path": tmp_path / "results",
        "scenarios_path": tmp_path / "scenarios",
    }
    run_info_kwargs[attribute_name] = invalid_name

    with pytest.raises(ValueError, match="must not contain path separators"):
        RunInfo(**run_info_kwargs)
