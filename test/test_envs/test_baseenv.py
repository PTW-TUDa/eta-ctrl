import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest

from eta_ctrl.common import episode_results_path
from eta_ctrl.config.config_run import ConfigRun
from eta_ctrl.envs.base_env import BaseEnv
from eta_ctrl.envs.state import StateConfig, StateVar
from examples.damped_oscillator.main import (
    experiment_conventional as ex_oscillator,
    get_path as get_oscillator_path,
)


class TestStateLog:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_oscillator_path()
        yield path
        shutil.rmtree(path / "results", ignore_errors=True)
        logging.shutdown()

    @pytest.fixture(scope="class")
    def results_path(self, experiment_path):
        return experiment_path / "results/conventional_series"

    @pytest.fixture(scope="class")
    def damped_oscillator_eta(self, experiment_path):
        mpl.use("Agg")  # Prevents GUI from opening
        return ex_oscillator(experiment_path, {"settings": {"log_to_file": False}})

    def test_export_state_log(self, damped_oscillator_eta, results_path):
        assert episode_results_path(results_path, "run1", 1, 1).exists()

    def test_export_with_datetime_index(self, damped_oscillator_eta, results_path):
        config = damped_oscillator_eta.config
        report = pd.read_csv(
            episode_results_path(results_path, "run1", 1, 1),
            sep=";",
            index_col=0,
        )
        report.index = pd.to_datetime(report.index)
        step = config.settings.sampling_time / config.settings.sim_steps_per_sample

        assert (report.index[1] - report.index[0]) == timedelta(seconds=step)


class TestBaseEnvStringRepresentations:
    """Test __str__ and __repr__ methods for BaseEnv."""

    @pytest.fixture(scope="class")
    def concrete_base_env(self):
        """Create a concrete BaseEnv subclass for testing."""

        class TestEnv(BaseEnv):
            @property
            def version(self):
                return "v2.1.0"

            @property
            def description(self):
                return "Test environment for string representation testing"

            def step(self, action):
                # Minimal implementation for testing
                self.n_steps += 1
                self.n_steps_longtime += 1
                return {}, 0.0, False, False, {}

            def close(self):
                pass

            def render(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_run = ConfigRun(
                series="test_series",
                name="repr_test_run",
                description="Test run for string representations",
                path_root=temp_path,
                path_results=temp_path / "results",
                path_scenarios=temp_path / "scenarios",
            )

            state_config = StateConfig(
                StateVar(name="heating_power", is_agent_action=True, low_value=0, high_value=5000),
                StateVar(name="cooling_power", is_agent_action=True, low_value=0, high_value=3000),
                StateVar(name="room_temp", is_agent_observation=True, low_value=15, high_value=30),
                StateVar(name="outside_temp", is_agent_observation=True, low_value=-20, high_value=40),
            )

            env = TestEnv(
                env_id=42,
                config_run=config_run,
                state_config=state_config,
                scenario_time_begin=datetime(2023, 6, 15, 8, 0),
                scenario_time_end=datetime(2023, 6, 15, 20, 0),
                episode_duration=7200,  # 2 hours
                sampling_time=300,  # 5 minutes
            )

            yield env

    def test_str_representation_initial_state(self, concrete_base_env):
        """Test __str__ method for initial state of BaseEnv."""
        # Reset to ensure initial state
        concrete_base_env.n_episodes = 0
        concrete_base_env.n_steps = 0

        expected = "TestEnv(id=42, 2 actions, 2 observations, Episode 0, Step 0/24)"
        assert str(concrete_base_env) == expected

    def test_str_representation_with_progress(self, concrete_base_env):
        """Test __str__ method after some episode progress."""
        concrete_base_env.n_episodes = 3
        concrete_base_env.n_steps = 15

        expected = "TestEnv(id=42, 2 actions, 2 observations, Episode 3, Step 15/24)"
        assert str(concrete_base_env) == expected

    def test_str_representation_different_env_sizes(self):
        """Test __str__ method with different numbers of actions and observations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_run = ConfigRun(
                series="test_series",
                name="size_test",
                description="Size test",
                path_root=temp_path,
                path_results=temp_path / "results",
            )

            # Create config with many actions and few observations
            many_actions_config = StateConfig(
                *[StateVar(name=f"action_{i}", is_agent_action=True, low_value=0, high_value=100) for i in range(8)],
                StateVar(name="single_obs", is_agent_observation=True, low_value=0, high_value=1),
            )

            class MinimalTestEnv(BaseEnv):
                @property
                def version(self):
                    return "v1.0.0"

                @property
                def description(self):
                    return "Minimal test env"

                def step(self, action):
                    pass

                def close(self):
                    pass

                def render(self):
                    pass

            env = MinimalTestEnv(
                env_id=1,
                config_run=config_run,
                state_config=many_actions_config,
                scenario_time_begin=datetime(2023, 1, 1),
                scenario_time_end=datetime(2023, 1, 2),
                episode_duration=3600,
                sampling_time=60,
            )

            expected = "MinimalTestEnv(id=1, 8 actions, 1 observations, Episode 0, Step 0/60)"
            assert str(env) == expected

    def test_repr_representation_initial_state(self, concrete_base_env):
        """Test __repr__ method for initial state of BaseEnv."""
        # Reset to ensure initial state
        concrete_base_env.n_episodes = 0
        concrete_base_env.n_steps = 0

        expected = (
            "TestEnv(env_id=42, run_name='repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(concrete_base_env) == expected

    def test_repr_representation_with_progress(self, concrete_base_env):
        """Test __repr__ method after some episode progress."""
        concrete_base_env.n_episodes = 10
        concrete_base_env.n_steps = 8

        expected = (
            "TestEnv(env_id=42, run_name='repr_test_run', n_episodes=10, n_steps=8, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(concrete_base_env) == expected

    def test_repr_representation_different_durations(self):
        """Test __repr__ method with different episode durations and sampling times."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_run = ConfigRun(
                series="duration_test_series",
                name="duration_test_run_with_long_name",
                description="Duration test",
                path_root=temp_path,
                path_results=temp_path / "results",
            )

            state_config = StateConfig(
                StateVar(name="test_action", is_agent_action=True),
                StateVar(name="test_obs", is_agent_observation=True),
            )

            class DurationTestEnv(BaseEnv):
                @property
                def version(self):
                    return "v3.2.1"

                @property
                def description(self):
                    return "Duration test environment"

                def step(self, action):
                    pass

                def close(self):
                    pass

                def render(self):
                    pass

            env = DurationTestEnv(
                env_id=999,
                config_run=config_run,
                state_config=state_config,
                scenario_time_begin=datetime(2023, 1, 1),
                scenario_time_end=datetime(2023, 1, 2),
                episode_duration=1800,  # 30 minutes
                sampling_time=30,  # 30 seconds
            )

            expected = (
                "DurationTestEnv(env_id=999, run_name='duration_test_run_with_long_name', "
                "n_episodes=0, n_steps=0, episode_duration=1800.0, sampling_time=30.0)"
            )
            assert repr(env) == expected

    def test_str_and_repr_consistency(self, concrete_base_env):
        """Test that __str__ and __repr__ provide different but consistent information."""
        str_result = str(concrete_base_env)
        repr_result = repr(concrete_base_env)

        # Should be different formats
        assert str_result != repr_result

        # Both should contain the class name
        assert "TestEnv" in str_result
        assert "TestEnv" in repr_result

        # Both should contain the env_id
        assert "42" in str_result
        assert "42" in repr_result

        # str should be more human-readable (shorter)
        assert len(str_result) < len(repr_result)

        # repr should contain more technical details
        assert "run_name" in repr_result
        assert "episode_duration" in repr_result
        assert "sampling_time" in repr_result
