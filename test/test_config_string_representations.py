"""Tests for __str__ and __repr__ of Config, ConfigRun, ConfigSettings, ConfigSetup, and EtaCtrl."""

import copy
import pathlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eta_ctrl.config import Config, ConfigRun, ConfigSettings, ConfigSetup
from eta_ctrl.core import EtaCtrl
from eta_ctrl.envs.state import StateConfig
from test.resources.config.config_python import config as python_dict

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SETUP_DICT = python_dict["setup"]
_SETTINGS_KEYS = ("episode_duration", "sampling_time", "n_episodes_play")


def _fresh_setup_dict(**overrides) -> dict:
    """Return a deep copy of the base setup dict with optional overrides."""
    d = copy.deepcopy(_SETUP_DICT)
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# TestConfigSetupStringRepresentations
# ---------------------------------------------------------------------------


class TestConfigSetupStringRepresentations:
    """Tests for ConfigSetup.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def config_setup(self):
        return ConfigSetup.from_dict(_fresh_setup_dict())

    # --- __str__ ---

    def test_str_shows_env_and_agent_names(self, config_setup):
        assert str(config_setup) == "ConfigSetup(env=PyomoEnv, agent=MpcAgent)"

    # --- __repr__ ---

    def test_repr_contains_full_import_paths(self, config_setup):
        result = repr(config_setup)
        assert "environment='eta_ctrl.envs.PyomoEnv'" in result
        assert "agent='eta_ctrl.agents.MpcAgent'" in result


# ---------------------------------------------------------------------------
# TestConfigSettingsStringRepresentations
# ---------------------------------------------------------------------------


class TestConfigSettingsStringRepresentations:
    """Tests for ConfigSettings.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def settings(self):
        return ConfigSettings(
            episode_duration=1800.0,
            sampling_time=10.0,
            n_environments=1,
            n_episodes_play=1,
            seed=123,
            verbose=2,
        )

    @pytest.fixture(scope="class")
    def settings_no_seed(self):
        return ConfigSettings(
            episode_duration=3600.0,
            sampling_time=60.0,
            n_environments=4,
            n_episodes_play=10,
        )

    # --- __str__ ---

    def test_str_contains_episode_duration(self, settings):
        assert "episode_duration=1800.0" in str(settings)

    def test_str_contains_sampling_time(self, settings):
        assert "sampling_time=10.0" in str(settings)

    def test_str_contains_n_environments(self, settings):
        assert "n_environments=1" in str(settings)

    def test_str_exact(self, settings):
        expected = "ConfigSettings(episode_duration=1800.0, sampling_time=10.0, n_environments=1)"
        assert str(settings) == expected

    # --- __repr__ ---

    def test_repr_contains_seed(self, settings):
        assert "seed=123" in repr(settings)

    def test_repr_contains_verbose(self, settings):
        assert "verbose=2" in repr(settings)

    def test_repr_exact(self, settings):
        expected = "ConfigSettings(episode_duration=1800.0, sampling_time=10.0, n_environments=1, seed=123, verbose=2)"
        assert repr(settings) == expected

    def test_repr_shows_none_seed_when_unset(self, settings_no_seed):
        assert "seed=None" in repr(settings_no_seed)


# ---------------------------------------------------------------------------
# TestConfigRunStringRepresentations
# ---------------------------------------------------------------------------


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

    # --- __repr__ ---

    def test_repr_contains_series_and_name(self, config_run):
        result = repr(config_run)
        assert "series='my_series'" in result
        assert "name='my_run'" in result

    def test_repr_contains_root_path(self, config_run):
        assert "root_path=" in repr(config_run)

    def test_repr_contains_results_path(self, config_run):
        assert "results_path=" in repr(config_run)


# ---------------------------------------------------------------------------
# TestConfigStringRepresentations
# ---------------------------------------------------------------------------


class TestConfigStringRepresentations:
    """Tests for Config.__str__ and __repr__."""

    @pytest.fixture(autouse=True)
    def prevent_state_config_loading(self, monkeypatch):
        class _DummyStateConfig:
            def __init__(self, source_file: Path):
                self.source_file = source_file

        monkeypatch.setattr(
            StateConfig,
            "from_file",
            lambda root_path, filename, **_: _DummyStateConfig(source_file=Path(root_path) / filename),
        )

    @pytest.fixture
    def config(self, config_resources_path):
        return Config._from_dict(
            config=copy.deepcopy(python_dict),
            config_name="test_config",
            root_path=config_resources_path,
        )

    # --- __str__ ---

    def test_str_contains_config_name(self, config):
        assert "test_config" in str(config)

    def test_str_contains_env_and_agent(self, config):
        result = str(config)
        assert "env=PyomoEnv" in result
        assert "agent=MpcAgent" in result

    def test_str_exact(self, config):
        assert str(config) == "Config 'test_config' (env=PyomoEnv, agent=MpcAgent)"

    # --- __repr__ ---

    def test_repr_contains_config_name(self, config):
        assert "config_name='test_config'" in repr(config)

    def test_repr_contains_results_relpath(self, config):
        assert "results_relpath='results'" in repr(config)

    def test_repr_contains_root_path(self, config):
        assert "root_path=" in repr(config)


# ---------------------------------------------------------------------------
# TestEtaCtrlStringRepresentations
# ---------------------------------------------------------------------------


class TestEtaCtrlStringRepresentations:
    """Tests for EtaCtrl.__str__ and __repr__ using mocked config to avoid file I/O."""

    @pytest.fixture(scope="class")
    def eta_ctrl_no_run(self):
        """EtaCtrl instance before prepare_run() has been called (config_run is None)."""
        env_cls = MagicMock()
        env_cls.__name__ = "DummyEnv"
        agent_cls = MagicMock()
        agent_cls.__name__ = "DummyAgent"

        ctrl = object.__new__(EtaCtrl)
        ctrl.config = MagicMock()
        ctrl.config.config_name = "unit_test_config"
        ctrl.config.setup.environment_class = env_cls
        ctrl.config.setup.agent_class = agent_cls
        ctrl.config.root_path = pathlib.Path("/mock/root")
        ctrl.config_run = None
        return ctrl

    @pytest.fixture(scope="class")
    def eta_ctrl_with_run(self, eta_ctrl_no_run):
        """EtaCtrl instance after prepare_run() has been called (config_run is set)."""
        eta_ctrl_no_run.config_run = MagicMock()
        return eta_ctrl_no_run

    # --- __str__ ---

    def test_str_exact(self, eta_ctrl_no_run):
        expected = "EtaCtrl(config='unit_test_config', env=DummyEnv, agent=DummyAgent)"
        assert str(eta_ctrl_no_run) == expected

    def test_str_contains_config_name(self, eta_ctrl_no_run):
        assert "unit_test_config" in str(eta_ctrl_no_run)

    # --- __repr__ ---

    def test_repr_contains_config_name(self, eta_ctrl_no_run):
        assert "config_name='unit_test_config'" in repr(eta_ctrl_no_run)

    def test_repr_shows_config_run_not_initialized(self, eta_ctrl_no_run):
        eta_ctrl_no_run.config_run = None
        assert "config_run_initialized=False" in repr(eta_ctrl_no_run)

    def test_repr_shows_config_run_initialized(self, eta_ctrl_with_run):
        assert "config_run_initialized=True" in repr(eta_ctrl_with_run)
