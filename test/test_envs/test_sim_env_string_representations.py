"""Tests for string representations of SimEnv class."""

from datetime import datetime

import pytest


class TestSimEnvStringRepresentations:
    """Test __str__ and __repr__ methods for SimEnv using unified factory."""

    @pytest.fixture(scope="class")
    def concrete_sim_env(self, unified_env_factory):
        """Create a concrete SimEnv instance for testing."""
        return unified_env_factory(
            env_type="sim",
            env_id=25,
            config_run_params={
                "series": "sim_test_series",
                "name": "sim_repr_test_run",
                "description": "Test run for SimEnv string representations",
            },
            state_config_type="sim",
            scenario_time_begin=datetime(2023, 3, 1, 12, 0),
            scenario_time_end=datetime(2023, 3, 1, 18, 0),
            episode_duration=3600,  # 1 hour
            sampling_time=60,  # 1 minute
            fmu_name="heating_model",
            sim_steps_per_sample=10,
        )

    def test_str_representation_initial_state(self, concrete_sim_env):
        """Test __str__ method for initial state of SimEnv."""
        # Reset to ensure initial state
        concrete_sim_env.n_episodes = 0
        concrete_sim_env.n_steps = 0

        expected = "TestSimEnv(id=25, 1 actions, 2 observations, Episode 0, Step 0/60), FMU: heating_model"
        assert str(concrete_sim_env) == expected

    def test_str_representation_with_progress(self, concrete_sim_env):
        """Test __str__ method after some episode progress."""
        concrete_sim_env.n_episodes = 2
        concrete_sim_env.n_steps = 30

        expected = "TestSimEnv(id=25, 1 actions, 2 observations, Episode 2, Step 30/60), FMU: heating_model"
        assert str(concrete_sim_env) == expected

    def test_str_representation_different_fmu(self, unified_env_factory):
        """Test __str__ method with different FMU name."""
        env = unified_env_factory(
            env_type="sim",
            env_id=15,
            config_run_params={
                "series": "fmu_test_series",
                "name": "fmu_test_run",
                "description": "FMU test",
            },
            state_config_type="basic",
            scenario_time_begin=datetime(2023, 1, 1, 0, 0),
            scenario_time_end=datetime(2023, 1, 1, 0, 20),
            episode_duration=1200,
            sampling_time=30,
            fmu_name="motor_dynamics_v2",
            sim_steps_per_sample=5,
        )
        env.n_episodes = 1
        env.n_steps = 5

        expected = "TestSimEnv(id=15, 1 actions, 1 observations, Episode 1, Step 5/40), FMU: motor_dynamics_v2"
        assert str(env) == expected

    def test_repr_representation_initial_state(self, concrete_sim_env):
        """Test __repr__ method for initial state of SimEnv."""
        # Reset to ensure initial state
        concrete_sim_env.n_episodes = 0
        concrete_sim_env.n_steps = 0

        expected = (
            "TestSimEnv(env_id=25, run_name='sim_repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=3600.0, sampling_time=60.0, fmu_name='heating_model', sim_steps_per_sample=10)"
        )
        assert repr(concrete_sim_env) == expected

    def test_repr_representation_with_progress(self, concrete_sim_env):
        """Test __repr__ method after some episode progress."""
        concrete_sim_env.n_episodes = 4
        concrete_sim_env.n_steps = 22

        expected = (
            "TestSimEnv(env_id=25, run_name='sim_repr_test_run', n_episodes=4, n_steps=22, "
            "episode_duration=3600.0, sampling_time=60.0, fmu_name='heating_model', sim_steps_per_sample=10)"
        )
        assert repr(concrete_sim_env) == expected
