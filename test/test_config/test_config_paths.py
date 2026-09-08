from pathlib import Path

import pytest

from eta_ctrl.config import ConfigPaths


class TestConfigPaths:
    # Simple class, simple tests
    def test_default_values(self):
        config_paths = ConfigPaths()
        assert config_paths.state_file_relpath is None
        assert config_paths.results_relpath == Path("results")
        assert config_paths.scenarios_relpath == Path("scenarios")
        assert config_paths.model_filename == Path("model.zip")
        assert config_paths.model_before_error_filename == Path("model_before_error.zip")
        assert config_paths.info_filename == Path("info.json")
        assert config_paths.log_output_filename == Path("log_output.log")
        assert config_paths.vec_normalize_filename == Path("vec_normalize.pkl")
        assert config_paths.net_arch_filename == Path("net_arch.txt")
        assert config_paths.models_relpath == Path("models")

    def test_all_values(self):
        config_extra_params = {
            "results_relpath": "results_foo",
            "scenarios_relpath": "scenarios_foo",
            "state_file_relpath": "config/test_env_state_config",
            "model_filename": "agent.model",
            "model_before_error_filename": "failed.model",
            "info_filename": "run.info",
            "log_output_filename": "run.log",
            "vec_normalize_filename": "normalization.pkl",
            "net_arch_filename": "architecture.txt",
            "models_relpath": "saved_models",
        }
        config_paths = ConfigPaths(**config_extra_params)
        assert config_paths.state_file_relpath == Path("config/test_env_state_config")
        assert config_paths.results_relpath == Path("results_foo")
        assert config_paths.scenarios_relpath == Path("scenarios_foo")
        assert config_paths.model_filename == Path("agent.model")
        assert config_paths.model_before_error_filename == Path("failed.model")
        assert config_paths.info_filename == Path("run.info")
        assert config_paths.log_output_filename == Path("run.log")
        assert config_paths.vec_normalize_filename == Path("normalization.pkl")
        assert config_paths.net_arch_filename == Path("architecture.txt")
        assert config_paths.models_relpath == Path("saved_models")

    def test_monitor_filename_still_accepted(self):
        with pytest.warns(DeprecationWarning, match=r"monitor_filename"):
            config_paths = ConfigPaths(monitor_filename="run.monitor")
        assert config_paths.monitor_filename == Path("run.monitor")

    def test_monitor_filename_default_does_not_warn(self):
        config_paths = ConfigPaths()
        assert config_paths.monitor_filename == Path("monitor.csv")
