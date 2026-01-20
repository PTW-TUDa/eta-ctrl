import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import gymnasium
import matplotlib as mpl
import numpy as np
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
    """Test __str__ and __repr__ methods for BaseEnv using unified factory."""

    @pytest.fixture(scope="class")
    def concrete_base_env(self, unified_env_factory):
        """Create a concrete BaseEnv instance for testing."""
        return unified_env_factory(
            env_type="base",
            env_id=42,
            config_run_params={
                "series": "test_series",
                "name": "repr_test_run",
                "description": "Test run for string representations",
            },
            state_config_type="default",
        )

    def test_str_representation_initial_state(self, concrete_base_env):
        """Test __str__ method for initial state of BaseEnv."""
        # Reset to ensure initial state
        concrete_base_env.n_episodes = 0
        concrete_base_env.n_steps = 0

        expected = "TestBaseEnv(id=42, 2 actions, 2 observations, Episode 0, Step 0/24)"
        assert str(concrete_base_env) == expected

    def test_str_representation_with_progress(self, concrete_base_env):
        """Test __str__ method after some episode progress."""
        concrete_base_env.n_episodes = 3
        concrete_base_env.n_steps = 15

        expected = "TestBaseEnv(id=42, 2 actions, 2 observations, Episode 3, Step 15/24)"
        assert str(concrete_base_env) == expected

    def test_str_representation_different_env_sizes(self, unified_env_factory):
        """Test __str__ method with different numbers of actions and observations."""
        env = unified_env_factory(
            env_type="base",
            env_id=1,
            config_run_params={"series": "test_series", "name": "size_test", "description": "Size test"},
            state_config_type="many_actions",
            scenario_time_begin=datetime(2023, 1, 1),
            scenario_time_end=datetime(2023, 1, 2),
            episode_duration=3600,
            sampling_time=60,
        )

        expected = "TestBaseEnv(id=1, 8 actions, 1 observations, Episode 0, Step 0/60)"
        assert str(env) == expected

    def test_repr_representation_initial_state(self, concrete_base_env):
        """Test __repr__ method for initial state of BaseEnv."""
        # Reset to ensure initial state
        concrete_base_env.n_episodes = 0
        concrete_base_env.n_steps = 0

        expected = (
            "TestBaseEnv(env_id=42, run_name='repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(concrete_base_env) == expected

    def test_repr_representation_with_progress(self, concrete_base_env):
        """Test __repr__ method after some episode progress."""
        concrete_base_env.n_episodes = 10
        concrete_base_env.n_steps = 8

        expected = (
            "TestBaseEnv(env_id=42, run_name='repr_test_run', n_episodes=10, n_steps=8, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(concrete_base_env) == expected

    def test_repr_representation_different_durations(self, unified_env_factory):
        """Test __repr__ method with different episode durations and sampling times."""
        env = unified_env_factory(
            env_type="base",
            env_id=999,
            config_run_params={
                "series": "duration_test_series",
                "name": "duration_test_run_with_long_name",
                "description": "Duration test",
            },
            state_config_type="minimal",
            scenario_time_begin=datetime(2023, 1, 1),
            scenario_time_end=datetime(2023, 1, 2),
            episode_duration=1800,  # 30 minutes
            sampling_time=30,  # 30 seconds
        )

        expected = (
            "TestBaseEnv(env_id=999, run_name='duration_test_run_with_long_name', "
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
        assert "TestBaseEnv" in str_result
        assert "TestBaseEnv" in repr_result

        # Both should contain the env_id
        assert "42" in str_result
        assert "42" in repr_result

        # str should be more human-readable (shorter)
        assert len(str_result) < len(repr_result)

        # repr should contain more technical details
        assert "run_name" in repr_result
        assert "episode_duration" in repr_result
        assert "sampling_time" in repr_result


class TestActionValidation:
    """Test the improved action validation error messages."""

    @pytest.fixture
    def test_env_factory(self):
        """Factory fixture to create test environments with different action spaces or state configs.

        Uses context manager to ensure temporary directories are properly cleaned up (no memory leaks).
        """

        def _create_env(action_space=None, state_config=None):
            """Create a test environment with optional custom action space or state config.

            Args:
                action_space: Optional custom action space to override default.
                state_config: Optional custom StateConfig. If not provided, uses default with one observation.

            Returns:
                Tuple of (env, temp_dir_context) where temp_dir_context must be kept alive.
            """

            class GenericTestEnv(BaseEnv):
                @property
                def version(self):
                    return "v1.0.0"

                @property
                def description(self):
                    return "Test environment for action validation"

                def _step(self):
                    return 0.0, False, False, {}

                def _reset(self, *, seed=None, options=None):
                    return {}

                def close(self):
                    pass

                def render(self):
                    pass

            # Use TemporaryDirectory context manager - automatically cleans up
            temp_dir_context = tempfile.TemporaryDirectory()
            temp_path = Path(temp_dir_context.name)

            config_run = ConfigRun(
                series="test_series",
                name="validation_test",
                description="Validation test",
                root_path=temp_path,
                results_path=temp_path,
                scenarios_path=temp_path,
            )

            # Use provided state_config or default
            if state_config is None:
                state_config = StateConfig(
                    StateVar(name="obs1", is_agent_observation=True, low_value=0, high_value=100),
                )

            env = GenericTestEnv(
                env_id=1,
                config_run=config_run,
                state_config=state_config,
                scenario_time_begin=datetime(2023, 1, 1),
                scenario_time_end=datetime(2023, 1, 2),
                episode_duration=3600,
                sampling_time=60,
            )

            # Override action space if provided
            if action_space is not None:
                env.action_space = action_space

            # Store cleanup context on env to prevent premature deletion
            env._temp_dir_context = temp_dir_context
            return env

        return _create_env

    @pytest.fixture
    def box_env(self, test_env_factory):
        """Create environment with Box action space."""
        box_state_config = StateConfig(
            StateVar(name="action1", is_agent_action=True, low_value=-1.0, high_value=2.0),
            StateVar(name="action2", is_agent_action=True, low_value=0.0, high_value=5.0),
            StateVar(name="action3", is_agent_action=True, low_value=-10.0, high_value=10.0),
            StateVar(name="obs1", is_agent_observation=True, low_value=0, high_value=100),
        )
        return test_env_factory(state_config=box_state_config)

    @pytest.fixture
    def discrete_env(self, test_env_factory):
        """Create environment with Discrete action space."""
        return test_env_factory(action_space=gymnasium.spaces.Discrete(5))

    @pytest.fixture
    def multi_discrete_env(self, test_env_factory):
        """Create environment with MultiDiscrete action space."""
        return test_env_factory(action_space=gymnasium.spaces.MultiDiscrete([5, 3, 10]))

    @pytest.fixture
    def dict_env(self, test_env_factory):
        """Create environment with Dict action space."""
        return test_env_factory(
            action_space=gymnasium.spaces.Dict(
                {
                    "position": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "velocity": gymnasium.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
                }
            )
        )

    # Box action tests
    def test_box_action_shape_mismatch(self, box_env):
        """Test error message for wrong action shape."""
        invalid_action = np.array([1.0, 2.0])

        with pytest.raises(RuntimeError) as exc_info:
            box_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Action validation failed" in error_msg
        assert "Shape mismatch" in error_msg
        assert "Expected: (3,)" in error_msg
        assert "Received: (2,)" in error_msg
        assert "Expected 3 action(s), but received 2 action(s)" in error_msg

    def test_box_action_out_of_bounds(self, box_env):
        """Test error message for actions outside bounds."""
        invalid_action = np.array([1.5, 10.0, -15.0])

        with pytest.raises(RuntimeError) as exc_info:
            box_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Bound violations" in error_msg
        assert "action[1] = 10" in error_msg
        assert "exceeds maximum bound of 5" in error_msg
        assert "action[2] = -15" in error_msg
        assert "is below minimum bound of -10" in error_msg

    def test_box_action_dtype_mismatch(self, box_env):
        """Test error message for wrong dtype."""
        invalid_action = np.array([1, 2, 3], dtype=np.int64)

        with pytest.raises(RuntimeError) as exc_info:
            box_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Action validation failed" in error_msg

    def test_box_action_multiple_violations(self, box_env):
        """Test error message with multiple bound violations."""
        invalid_action = np.array([5.0, 10.0, 20.0])

        with pytest.raises(RuntimeError) as exc_info:
            box_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Bound violations" in error_msg
        assert "action[0]" in error_msg
        assert "action[1]" in error_msg
        assert "action[2]" in error_msg

    def test_valid_box_action_no_error(self, box_env):
        """Test that valid actions don't raise errors."""
        valid_action = np.array([0.5, 2.5, 5.0], dtype=np.float32)
        box_env._actions_valid(valid_action)

    def test_format_array_truncation(self, box_env):
        """Test that large arrays are formatted with truncation."""
        large_action = np.ones(100) * 5.5
        formatted = box_env._format_array(large_action, max_items=10)

        assert "..." in formatted
        assert "shape:" in formatted
        assert "(100,)" in formatted
        assert "dtype:" in formatted

    def test_error_message_includes_action_space(self, box_env):
        """Test that error messages include the action space description."""
        invalid_action = np.array([100.0, 100.0, 100.0])

        with pytest.raises(RuntimeError) as exc_info:
            box_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Action space:" in error_msg
        assert "Box" in error_msg

    # Discrete action tests
    def test_discrete_action_out_of_range_high(self, discrete_env):
        """Test error message for discrete action value too high."""
        invalid_action = np.array([10])

        with pytest.raises(RuntimeError) as exc_info:
            discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Value out of range" in error_msg
        assert "Valid range: [0, 4]" in error_msg
        assert "Received: 10" in error_msg

    def test_discrete_action_out_of_range_negative(self, discrete_env):
        """Test error message for negative discrete action."""
        invalid_action = np.array([-1])

        with pytest.raises(RuntimeError) as exc_info:
            discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Value out of range" in error_msg
        assert "Received: -1" in error_msg

    def test_discrete_action_wrong_shape(self, discrete_env):
        """Test error message for discrete action with wrong shape."""
        invalid_action = np.array([1, 2, 3])

        with pytest.raises(RuntimeError) as exc_info:
            discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Shape mismatch" in error_msg

    def test_discrete_action_valid(self, discrete_env):
        """Test that valid discrete actions pass."""
        discrete_env._actions_valid(3)

    # MultiDiscrete action tests
    def test_multi_discrete_shape_mismatch(self, multi_discrete_env):
        """Test error message for wrong shape in MultiDiscrete."""
        invalid_action = np.array([1, 2])

        with pytest.raises(RuntimeError) as exc_info:
            multi_discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Shape mismatch" in error_msg
        assert "Expected: (3,)" in error_msg
        assert "Received: (2,)" in error_msg

    def test_multi_discrete_value_violations(self, multi_discrete_env):
        """Test error message for out-of-range values in MultiDiscrete."""
        invalid_action = np.array([2, 5, 15])

        with pytest.raises(RuntimeError) as exc_info:
            multi_discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Value violations" in error_msg
        assert "action[1]" in error_msg
        assert "action[2]" in error_msg

    def test_multi_discrete_single_violation(self, multi_discrete_env):
        """Test error message for single violation in MultiDiscrete."""
        invalid_action = np.array([0, 1, 20])

        with pytest.raises(RuntimeError) as exc_info:
            multi_discrete_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Value violations (1 found)" in error_msg
        assert "action[2] = 20" in error_msg
        assert "outside valid range [0, 9]" in error_msg

    def test_multi_discrete_valid(self, multi_discrete_env):
        """Test that valid MultiDiscrete actions pass."""
        multi_discrete_env._actions_valid(np.array([4, 2, 9]))

    # Dict action tests
    def test_dict_action_wrong_type(self, dict_env):
        """Test error message when providing array instead of dict."""
        invalid_action = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(RuntimeError) as exc_info:
            dict_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Dict action space validation failed" in error_msg
        assert "Expected a dictionary with keys:" in error_msg
        assert "position" in error_msg
        assert "velocity" in error_msg
        assert "Received type: ndarray" in error_msg

    def test_dict_action_missing_keys(self, dict_env):
        """Test error message for missing keys in dict action."""
        invalid_action = {"position": np.array([0.5, 0.5], dtype=np.float32)}

        with pytest.raises(RuntimeError) as exc_info:
            dict_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Missing keys:" in error_msg
        assert "velocity" in error_msg

    def test_dict_action_extra_keys(self, dict_env):
        """Test error message for extra keys in dict action."""
        invalid_action = {
            "position": np.array([0.5, 0.5], dtype=np.float32),
            "velocity": np.array([1.0, 2.0], dtype=np.float32),
            "acceleration": np.array([0.1, 0.2], dtype=np.float32),
        }

        with pytest.raises(RuntimeError) as exc_info:
            dict_env._actions_valid(invalid_action)

        error_msg = str(exc_info.value)
        assert "Unexpected keys:" in error_msg
        assert "acceleration" in error_msg


class TestUnifiedEnvironmentFactory:
    """Test the unified environment factory to ensure it creates all environment types correctly."""

    def test_create_base_env(self, unified_env_factory):
        """Test creating BaseEnv using unified factory."""
        env = unified_env_factory(env_type="base", env_id=1, state_config_type="basic")
        assert env.__class__.__name__ == "TestBaseEnv"
        assert env.env_id == 1
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_create_pyomo_env(self, unified_env_factory):
        """Test creating PyomoEnv using unified factory."""
        env = unified_env_factory(
            env_type="pyomo",
            env_id=2,
            state_config_type="multi_action",
            n_prediction_steps=24,
            prediction_horizon=7200.0,
        )
        assert env.__class__.__name__ == "TestPyomoEnv"
        assert env.env_id == 2
        assert env.n_prediction_steps == 24
        assert env.prediction_horizon == 7200.0
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_create_sim_env(self, unified_env_factory):
        """Test creating SimEnv using unified factory."""
        env = unified_env_factory(
            env_type="sim", env_id=3, state_config_type="sim", fmu_name="custom_model.fmu", sim_steps_per_sample=5
        )
        assert env.__class__.__name__ == "TestSimEnv"
        assert env.env_id == 3
        assert env.fmu_name == "custom_model.fmu"
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_create_live_env(self, unified_env_factory):
        """Test creating LiveEnv using unified factory."""
        env = unified_env_factory(
            env_type="live", env_id=4, state_config_type="live", config_name="production_config", max_errors=50
        )
        assert env.__class__.__name__ == "TestLiveEnv"
        assert env.env_id == 4
        assert env.config_name == "production_config"
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_factory_error_handling(self, unified_env_factory):
        """Test that factory properly handles invalid environment types."""
        with pytest.raises(ValueError, match="Unknown env_type: invalid"):
            unified_env_factory(env_type="invalid")

    def test_state_config_factory_integration(self, unified_env_factory):
        """Test that factory correctly uses different state config types."""
        # Test all state config types
        for config_type in ["default", "many_actions", "minimal", "basic", "multi_action", "sim", "live"]:
            env = unified_env_factory(env_type="base", state_config_type=config_type)
            assert env.state_config is not None
            # Verify action and observation spaces are created
            assert hasattr(env, "action_space")
            assert hasattr(env, "observation_space")
