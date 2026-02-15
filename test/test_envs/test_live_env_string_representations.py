"""Tests for string representations of LiveEnv class."""

from datetime import datetime

import pytest


class TestLiveEnvStringRepresentations:
    """Test __str__ and __repr__ methods for LiveEnv using unified factory."""

    @pytest.fixture(scope="class")
    def concrete_live_env(self, unified_env_factory):
        """Create a concrete LiveEnv instance for testing."""
        return unified_env_factory(
            env_type="live",
            env_id=88,
            config_run_params={
                "series": "live_test_series",
                "name": "live_repr_test_run",
                "description": "Test run for LiveEnv string representations",
            },
            state_config_type="live",
            scenario_time_begin=datetime(2023, 9, 10, 9, 0),
            scenario_time_end=datetime(2023, 9, 10, 17, 0),
            episode_duration=1800,  # 30 minutes
            sampling_time=10,  # 10 seconds
            config_name="production_controller_config",
            max_errors=25,
        )

    def test_str_representation_initial_state(self, concrete_live_env):
        """Test __str__ method for initial state of LiveEnv."""
        # Reset to ensure initial state
        concrete_live_env.n_episodes = 0
        concrete_live_env.n_steps = 0

        expected = (
            "TestLiveEnv(id=88, 1 actions, 2 observations, Episode 0, Step 0/180), "
            "Live config: production_controller_config"
        )
        assert str(concrete_live_env) == expected

    def test_str_representation_with_progress(self, concrete_live_env):
        """Test __str__ method after some episode progress."""
        concrete_live_env.n_episodes = 1
        concrete_live_env.n_steps = 45

        expected = (
            "TestLiveEnv(id=88, 1 actions, 2 observations, Episode 1, Step 45/180), "
            "Live config: production_controller_config"
        )
        assert str(concrete_live_env) == expected

    def test_str_representation_different_config(self, unified_env_factory):
        """Test __str__ method with different config name."""
        env = unified_env_factory(
            env_type="live",
            env_id=77,
            config_run_params={
                "series": "config_test_series",
                "name": "config_test_run",
                "description": "Config test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 0, 10),
            episode_duration=600,
            sampling_time=5,
            config_name="development_test_config",
            max_errors=15,
        )
        env.n_episodes = 0
        env.n_steps = 10

        expected = (
            "TestLiveEnv(id=77, 1 actions, 1 observations, Episode 0, Step 10/120), "
            "Live config: development_test_config"
        )
        assert str(env) == expected

    def test_repr_representation_initial_state(self, concrete_live_env):
        """Test __repr__ method for initial state of LiveEnv."""
        # Reset to ensure initial state
        concrete_live_env.n_episodes = 0
        concrete_live_env.n_steps = 0

        expected = (
            "TestLiveEnv(env_id=88, run_name='live_repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=1800.0, sampling_time=10.0, "
            "config_name='production_controller_config', max_error_count=25)"
        )
        assert repr(concrete_live_env) == expected

    def test_repr_representation_with_progress(self, concrete_live_env):
        """Test __repr__ method after some episode progress."""
        concrete_live_env.n_episodes = 3
        concrete_live_env.n_steps = 120

        expected = (
            "TestLiveEnv(env_id=88, run_name='live_repr_test_run', n_episodes=3, n_steps=120, "
            "episode_duration=1800.0, sampling_time=10.0, "
            "config_name='production_controller_config', max_error_count=25)"
        )
        assert repr(concrete_live_env) == expected
