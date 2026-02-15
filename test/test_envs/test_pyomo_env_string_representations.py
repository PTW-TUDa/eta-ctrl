"""Tests for string representations of PyomoEnv class."""

from datetime import datetime

import pytest


class TestPyomoEnvStringRepresentations:
    """Test __str__ and __repr__ methods for PyomoEnv using unified factory."""

    @pytest.fixture(scope="class")
    def concrete_pyomo_env(self, unified_env_factory):
        """Create a concrete PyomoEnv instance for testing."""
        return unified_env_factory(
            env_type="pyomo",
            env_id=42,
            config_run_params={
                "series": "pyomo_test_series",
                "name": "pyomo_repr_test_run",
                "description": "Test run for PyomoEnv string representations",
            },
            state_config_type="multi_action",
            scenario_time_begin=datetime(2023, 6, 15, 8, 0),
            scenario_time_end=datetime(2023, 6, 15, 20, 0),
            episode_duration=7200,  # 2 hours
            sampling_time=300,  # 5 minutes
            model_parameters={},  # Empty dict for testing
            prediction_horizon=3600.0,  # 1 hour
            n_prediction_steps=12,  # 12 prediction steps
        )

    def test_str_representation_initial_state(self, concrete_pyomo_env):
        """Test __str__ method for initial state of PyomoEnv."""
        # Reset to ensure initial state
        concrete_pyomo_env.n_episodes = 0
        concrete_pyomo_env.n_steps = 0

        expected = "TestPyomoEnv(id=42, 2 actions, 2 observations, Episode 0, Step 0/24), Prediction steps: 12"
        assert str(concrete_pyomo_env) == expected

    def test_str_representation_with_progress(self, concrete_pyomo_env):
        """Test __str__ method after some episode progress."""
        concrete_pyomo_env.n_episodes = 3
        concrete_pyomo_env.n_steps = 15

        expected = "TestPyomoEnv(id=42, 2 actions, 2 observations, Episode 3, Step 15/24), Prediction steps: 12"
        assert str(concrete_pyomo_env) == expected

    def test_str_representation_different_prediction_steps(self, unified_env_factory):
        """Test __str__ method with different prediction steps."""
        env = unified_env_factory(
            env_type="pyomo",
            env_id=10,
            config_run_params={
                "series": "prediction_test_series",
                "name": "prediction_test_run",
                "description": "Prediction test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 1, 0),
            episode_duration=3600,
            sampling_time=60,
            model_parameters={},
            prediction_horizon=1800.0,
            n_prediction_steps=30,
        )
        env.n_episodes = 0
        env.n_steps = 0

        expected = "TestPyomoEnv(id=10, 1 actions, 1 observations, Episode 0, Step 0/60), Prediction steps: 30"
        assert str(env) == expected

    def test_repr_representation_initial_state(self, concrete_pyomo_env):
        """Test __repr__ method for initial state of PyomoEnv."""
        # Reset to ensure initial state
        concrete_pyomo_env.n_episodes = 0
        concrete_pyomo_env.n_steps = 0

        expected = (
            "TestPyomoEnv(env_id=42, run_name='pyomo_repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=7200.0, sampling_time=300.0, prediction_horizon=3600.0, n_prediction_steps=12)"
        )
        assert repr(concrete_pyomo_env) == expected

    def test_repr_representation_with_progress(self, concrete_pyomo_env):
        """Test __repr__ method after some episode progress."""
        concrete_pyomo_env.n_episodes = 5
        concrete_pyomo_env.n_steps = 18

        expected = (
            "TestPyomoEnv(env_id=42, run_name='pyomo_repr_test_run', n_episodes=5, n_steps=18, "
            "episode_duration=7200.0, sampling_time=300.0, prediction_horizon=3600.0, n_prediction_steps=12)"
        )
        assert repr(concrete_pyomo_env) == expected

    def test_repr_representation_different_parameters(self, unified_env_factory):
        """Test __repr__ method with different prediction parameters."""
        env = unified_env_factory(
            env_type="pyomo",
            env_id=999,
            config_run_params={
                "series": "different_params_series",
                "name": "different_params_run",
                "description": "Different parameters test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 0, 30),
            episode_duration=1800.0,
            sampling_time=30.0,
            model_parameters={},
            prediction_horizon=900.0,
            n_prediction_steps=30,
        )
        env.n_episodes = 2
        env.n_steps = 45

        expected = (
            "TestPyomoEnv(env_id=999, run_name='different_params_run', "
            "n_episodes=2, n_steps=45, episode_duration=1800.0, sampling_time=30.0, "
            "prediction_horizon=900.0, n_prediction_steps=30)"
        )
        assert repr(env) == expected
