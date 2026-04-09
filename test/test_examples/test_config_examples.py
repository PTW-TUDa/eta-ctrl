import pytest

from eta_ctrl.envs.state import StateConfig
from eta_ctrl.timeseries.scenario_manager import ConfigCsvScenario, CsvScenarioManager
from examples.config.main import load_full, load_minimal, load_minimal_with_scenarios


@pytest.fixture(autouse=True)
def _patch_file_loading(monkeypatch):
    """Prevent actual file loading (state config, scenario CSVs) during config construction."""
    monkeypatch.setattr(StateConfig, "from_file", lambda **_kwargs: StateConfig())
    monkeypatch.setattr(ConfigCsvScenario, "model_post_init", lambda *_args: None)
    monkeypatch.setattr(CsvScenarioManager, "load_data", lambda *_args: None)


class TestConfigExamples:
    def test_minimal(self):
        load_minimal()

    def test_minimal_with_scenarios(self):
        config = load_minimal_with_scenarios()
        assert config.settings.scenario_files is not None

    def test_full(self):
        load_full()
