"""Integration tests for environment string representations across all environment types."""

import inspect
from datetime import datetime

from eta_ctrl.envs.live_env import LiveEnv
from eta_ctrl.envs.pyomo_env import PyomoEnv
from eta_ctrl.envs.sim_env import SimEnv


class TestEnvironmentStringRepresentationIntegration:
    """Integration tests for environment string representations."""

    def test_method_implementations_exist(self):
        """Test that all environment classes have the string methods implemented."""
        # Test that methods exist and are not just inherited from object

        # Check that the methods exist
        assert hasattr(PyomoEnv, "__str__")
        assert hasattr(PyomoEnv, "__repr__")
        assert hasattr(SimEnv, "__str__")
        assert hasattr(SimEnv, "__repr__")
        assert hasattr(LiveEnv, "__str__")
        assert hasattr(LiveEnv, "__repr__")

        # Check that they're actually implemented (not just object.__str__)
        assert PyomoEnv.__str__ is not object.__str__
        assert PyomoEnv.__repr__ is not object.__repr__
        assert SimEnv.__str__ is not object.__str__
        assert SimEnv.__repr__ is not object.__repr__
        assert LiveEnv.__str__ is not object.__str__
        assert LiveEnv.__repr__ is not object.__repr__

    def test_string_method_signatures(self):
        """Test that the string methods have correct signatures."""
        # Test that methods can be called (they should exist)

        # Check __str__ methods
        assert len(inspect.signature(PyomoEnv.__str__).parameters) == 1  # just self
        assert len(inspect.signature(SimEnv.__str__).parameters) == 1
        assert len(inspect.signature(LiveEnv.__str__).parameters) == 1

        # Check __repr__ methods
        assert len(inspect.signature(PyomoEnv.__repr__).parameters) == 1
        assert len(inspect.signature(SimEnv.__repr__).parameters) == 1
        assert len(inspect.signature(LiveEnv.__repr__).parameters) == 1

    def test_str_and_repr_consistency_pattern(self, unified_env_factory):
        """Test that __str__ and __repr__ follow consistent patterns across all environments."""
        # This test verifies that all environment types follow the same pattern

        # Create environment instances using the unified factory
        pyomo_env = unified_env_factory(
            env_type="pyomo",
            env_id=1,
            config_run_params={
                "series": "consistency_test",
                "name": "consistency_run",
                "description": "Consistency test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 1, 0),
            episode_duration=3600,
            sampling_time=60,
            model_parameters={},
            prediction_horizon=1800.0,
            n_prediction_steps=10,
        )

        sim_env = unified_env_factory(
            env_type="sim",
            env_id=1,
            config_run_params={
                "series": "consistency_test",
                "name": "consistency_run",
                "description": "Consistency test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 1, 0),
            episode_duration=3600,
            sampling_time=60,
            fmu_name="test_model",
            sim_steps_per_sample=5,
        )

        live_env = unified_env_factory(
            env_type="live",
            env_id=1,
            config_run_params={
                "series": "consistency_test",
                "name": "consistency_run",
                "description": "Consistency test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 1, 0),
            episode_duration=3600,
            sampling_time=60,
            config_name="test_config",
            max_errors=10,
        )

        # Verify each has the base pattern plus specific info
        assert "Episode 0, Step 0/60" in str(pyomo_env)
        assert "Prediction steps:" in str(pyomo_env)

        assert "Episode 0, Step 0/60" in str(sim_env)
        assert "FMU:" in str(sim_env)

        assert "Episode 0, Step 0/60" in str(live_env)
        assert "Live config:" in str(live_env)

        # Verify repr patterns
        assert "run_name='consistency_run'" in repr(pyomo_env)
        assert "episode_duration=3600" in repr(pyomo_env)

        assert "run_name='consistency_run'" in repr(sim_env)
        assert "episode_duration=3600" in repr(sim_env)

        assert "run_name='consistency_run'" in repr(live_env)
        assert "episode_duration=3600" in repr(live_env)
