import pytest

import eta_ctrl.envs as envs_module


class TestEnvsLazyImports:
    """Test that __getattr__ lazy-loading in eta_ctrl.envs works correctly."""

    def test_live_env_is_accessible(self) -> None:
        cls = envs_module.LiveEnv
        assert cls is not None
        assert cls.__name__ == "LiveEnv"

    def test_no_vec_env_is_accessible(self) -> None:
        cls = envs_module.NoVecEnv
        assert cls is not None
        assert cls.__name__ == "NoVecEnv"

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match=r"module 'eta_ctrl\.envs' has no attribute 'NonExistent'"):
            _ = envs_module.NonExistent
