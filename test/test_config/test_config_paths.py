from pathlib import Path

from eta_ctrl.config import ConfigPaths


class TestConfigPaths:
    # Simple class, simple tests
    def test_default_values(self):
        config_paths = ConfigPaths()
        assert config_paths.state_file_relpath is None
        assert config_paths.results_relpath == Path("results")
        assert config_paths.scenarios_relpath == Path("scenarios")

    def test_all_values(self):
        config_extra_params = {
            "results_relpath": "results_foo",
            "scenarios_relpath": "scenarios_foo",
            "state_file_relpath": "config/test_env_state_config",
        }
        config_paths = ConfigPaths(**config_extra_params)
        assert config_paths.state_file_relpath == Path("config/test_env_state_config")
        assert config_paths.results_relpath == Path("results_foo")
        assert config_paths.scenarios_relpath == Path("scenarios_foo")
