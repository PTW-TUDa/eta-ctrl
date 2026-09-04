import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import gymnasium
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from eta_ctrl.common import episode_results_path
from eta_ctrl.envs.base_env import BaseEnv
from examples.damped_oscillator.main import (
    experiment_conventional as ex_oscillator,
    get_path as get_oscillator_path,
)


class TestStateLog:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        return get_oscillator_path()

    @pytest.fixture(scope="class")
    def results_root(self, tmp_path_factory):
        return tmp_path_factory.mktemp("damped_oscillator_results") / "results"

    @pytest.fixture(scope="class")
    def results_path(self, results_root):
        return results_root / "conventional_series"

    @pytest.fixture(scope="class")
    def damped_oscillator_eta(self, experiment_path, results_root):
        mpl.use("Agg")  # Prevents GUI from opening
        try:
            return ex_oscillator(
                experiment_path,
                {
                    "paths": {"results_relpath": results_root},
                    "settings": {"log_to_file": False},
                },
            )
        finally:
            logging.shutdown()

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
    """Tests for BaseEnv __str__ and __repr__."""

    @pytest.fixture(scope="class")
    def base_env(self, unified_env_factory):
        return unified_env_factory(
            env_type="base",
            env_id=42,
            run_info_params={
                "series": "test_series",
                "name": "repr_test_run",
                "description": "Test run for string representations",
            },
            state_config_type="default",
        )

    def test_str_representation(self, base_env):
        base_env.n_episodes = 0
        base_env.n_steps = 0
        expected = "TestBaseEnv(id=42, 2 actions, 2 observations, Episode 0, Step 0/24)"
        assert str(base_env) == expected

    def test_str_representation_after_step_progress(self, base_env):
        base_env.n_episodes = 3
        base_env.n_steps = 15
        expected = "TestBaseEnv(id=42, 2 actions, 2 observations, Episode 3, Step 15/24)"
        assert str(base_env) == expected

    def test_str_representation_env_sizes(self, unified_env_factory):
        env = unified_env_factory(
            env_type="base",
            env_id=1,
            run_info_params={"series": "test_series", "name": "size_test", "description": "Size test"},
            state_config_type="many_actions",
            scenario_time_begin=datetime(2023, 1, 1),
            scenario_time_end=datetime(2023, 1, 2),
            episode_duration=3600,
            sampling_time=60,
        )

        expected = "TestBaseEnv(id=1, 8 actions, 1 observations, Episode 0, Step 0/60)"
        assert str(env) == expected

    def test_repr_representation(self, base_env):
        base_env.n_episodes = 0
        base_env.n_steps = 0
        expected = (
            "TestBaseEnv(env_id=42, run_name='repr_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(base_env) == expected

    def test_repr_representation_after_step_progress(self, base_env):
        base_env.n_episodes = 10
        base_env.n_steps = 8
        expected = (
            "TestBaseEnv(env_id=42, run_name='repr_test_run', n_episodes=10, n_steps=8, "
            "episode_duration=7200.0, sampling_time=300.0)"
        )
        assert repr(base_env) == expected

    def test_repr_representation_different_durations(self, unified_env_factory):
        env = unified_env_factory(
            env_type="base",
            env_id=999,
            run_info_params={
                "series": "duration_test_series",
                "name": "duration_test_run_with_long_name",
                "description": "Duration test",
            },
            state_config_type="minimal",
            scenario_time_begin=datetime(2023, 1, 1),
            scenario_time_end=datetime(2023, 1, 2),
            episode_duration=1800,
            sampling_time=30,
        )

        expected = (
            "TestBaseEnv(env_id=999, run_name='duration_test_run_with_long_name', "
            "n_episodes=0, n_steps=0, episode_duration=1800.0, sampling_time=30.0)"
        )
        assert repr(env) == expected

    def test_str_and_repr_consistency(self, base_env):
        str_result = str(base_env)
        repr_result = repr(base_env)
        assert str_result != repr_result
        assert "TestBaseEnv" in str_result
        assert "TestBaseEnv" in repr_result
        assert "42" in str_result
        assert "42" in repr_result
        assert len(str_result) < len(repr_result)
        assert "run_name" in repr_result
        assert "episode_duration" in repr_result
        assert "sampling_time" in repr_result


class TestBaseEnvPublicMethods:
    """Method-focused tests for BaseEnv public Methods."""

    @pytest.fixture
    def method_env(self, unified_env_factory):
        return unified_env_factory(
            env_type="method",
            env_id=1,
            run_info_params={"series": "api_series", "name": "api_run", "description": "api tests"},
            state_config_type="method_test",
            episode_duration=3600,
            sampling_time=60,
        )

    def test_get_info(self):
        class _ClassAttrEnv(BaseEnv):
            version = "v1.0.0"
            description = "BaseEnv method test env"

            def _step(self):
                return 0.0, False, False, {}

            def _reset(self, *, seed=None, options=None):
                return {}

            def close(self):
                pass

            def render(self):
                pass

        assert _ClassAttrEnv.get_info() == ("v1.0.0", "BaseEnv method test env")

    def test_get_observations_success(self, method_env):
        method_env.state = {"obs": np.array([3.0])}
        observations = method_env.get_observations()
        assert observations["obs"][0] == 3.0

    def test_get_observations_missing_key(self, method_env):
        method_env.state = {}
        with pytest.raises(KeyError, match="unavailable in environment state"):
            method_env.get_observations()

    def test_set_action_array_and_dict(self, method_env):
        method_env.state = {}
        method_env.set_action(np.array([4.0], dtype=np.float32))
        assert method_env.state["act"][0] == 4.0

        method_env.set_action({"act": np.array([7.0])})
        assert method_env.state["act"][0] == 7.0

    def test_set_external_outputs_and_missing_key(self, method_env):
        method_env.state = {}
        method_env.set_external_outputs({"ext.out": 2.0})
        assert np.isclose(method_env.state["ext_out_state"][0], 9.0)

        with pytest.raises(KeyError, match="Missing value for external output"):
            method_env.set_external_outputs({})

    def test_public_reset_and_step(self, method_env):
        observations, info = method_env.reset(seed=42)
        assert observations["obs"][0] == 5.0
        assert info == {"reset": True}

        observations, reward, terminated, truncated, info = method_env.step(np.array([6.0], dtype=np.float32))
        assert observations["obs"][0] == 6.0
        assert reward == 1.0
        assert not terminated
        assert not truncated
        assert info == {"source": "_step"}

    def test_set_scenario_state_reset_updates_offset_and_state(self, method_env):
        method_env.state = {}
        method_env.n_steps = 3
        method_env._scenario_offset = 0
        method_env.set_scenario_state(reset=True)

        assert method_env._scenario_offset == 2
        assert "scen" in method_env.state

    def test_get_external_inputs_returns_ext_ids_as_keys(self, method_env):
        method_env.state = {"ext_in_state": np.array([30.0])}
        external_inputs = method_env.get_external_inputs()
        assert set(external_inputs) == {"ext.in"}
        # Value de-scaled: 30.0 / 2.0 - 10.0 = 5.0
        assert np.isclose(external_inputs["ext.in"], 5.0)
        assert "ext_in_state" not in external_inputs

    def test_get_external_inputs_correct_value_scaling(self, method_env):
        method_env.state = {"ext_in_state": np.array([42.0])}
        external_inputs = method_env.get_external_inputs()
        # Value de-scaled: 42.0 / 2.0 - 10.0 = 11.0
        assert external_inputs == {"ext.in": 11.0}

    def test_get_external_inputs_missing_key(self, method_env):
        method_env.state = {}
        with pytest.raises(KeyError, match="unavailable in environment state"):
            method_env.get_external_inputs()


class TestActionValidation:
    """Tests for _actions_valid and detailed validation messages."""

    @pytest.fixture
    def validation_env(self, unified_env_factory):
        return unified_env_factory(
            env_type="method",
            env_id=2,
            run_info_params={"series": "validation_series", "name": "validation_run", "description": "validate"},
            state_config_type="validation",
            episode_duration=3600,
            sampling_time=60,
        )

    def test_box_action_shape_and_bounds_errors(self, validation_env):
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            validation_env._actions_valid(np.array([1.0, 2.0]))

        with pytest.raises(RuntimeError, match="Bound violations"):
            validation_env._actions_valid(np.array([1.5, 10.0, -15.0]))

    def test_valid_box_action(self, validation_env):
        validation_env._actions_valid(np.array([0.5, 2.5, 5.0], dtype=np.float32))

    def test_format_array_truncation(self, validation_env):
        formatted = validation_env._format_array(np.ones(100) * 5.5, max_items=10)
        assert "..." in formatted
        assert "shape:" in formatted
        assert "(100,)" in formatted
        assert "dtype:" in formatted

    @pytest.mark.parametrize("invalid_action", [np.array([10]), np.array([-1])])
    def test_discrete_action_out_of_range(self, validation_env, invalid_action):
        validation_env.action_space = gymnasium.spaces.Discrete(5)
        with pytest.raises(RuntimeError, match="Value out of range"):
            validation_env._actions_valid(invalid_action)

    def test_discrete_action_wrong_shape(self, validation_env):
        validation_env.action_space = gymnasium.spaces.Discrete(5)
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            validation_env._actions_valid(np.array([1, 2, 3]))

    def test_discrete_action_valid(self, validation_env):
        validation_env.action_space = gymnasium.spaces.Discrete(5)
        validation_env._actions_valid(3)

    def test_multi_discrete_validation(self, validation_env):
        validation_env.action_space = gymnasium.spaces.MultiDiscrete([5, 3, 10])

        with pytest.raises(RuntimeError, match="Shape mismatch"):
            validation_env._actions_valid(np.array([1, 2]))

        with pytest.raises(RuntimeError, match="Value violations"):
            validation_env._actions_valid(np.array([2, 5, 15]))

        validation_env._actions_valid(np.array([4, 2, 9]))

    def test_dict_action_validation(self, validation_env):
        validation_env.action_space = gymnasium.spaces.Dict(
            {
                "position": gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "velocity": gymnasium.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )

        with pytest.raises(RuntimeError, match="Dict action space validation failed"):
            validation_env._actions_valid(np.array([1.0, 2.0, 3.0, 4.0]))

        with pytest.raises(RuntimeError, match="Missing keys"):
            validation_env._actions_valid({"position": np.array([0.5, 0.5], dtype=np.float32)})

        with pytest.raises(RuntimeError, match="Unexpected keys"):
            validation_env._actions_valid(
                {
                    "position": np.array([0.5, 0.5], dtype=np.float32),
                    "velocity": np.array([1.0, 2.0], dtype=np.float32),
                    "acceleration": np.array([0.1, 0.2], dtype=np.float32),
                }
            )


class TestUnifiedEnvironmentFactory:
    """Tests for shared unified env factory."""

    def test_create_base_env_type(self, unified_env_factory):
        env = unified_env_factory(env_type="base", env_id=1, state_config_type="basic")
        assert env.__class__.__name__ == "TestBaseEnv"
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

    def test_factory_error_handling(self, unified_env_factory):
        with pytest.raises(ValueError, match="Unknown env_type: invalid"):
            unified_env_factory(env_type="invalid")

    def test_state_config_factory_integration(self, unified_env_factory):
        for config_type in ["default", "many_actions", "minimal", "basic", "multi_action", "sim", "live"]:
            env = unified_env_factory(env_type="base", state_config_type=config_type)
            assert env.state_config is not None
            assert hasattr(env, "action_space")
            assert hasattr(env, "observation_space")


class TestPathEnvResilience:
    """Test that BaseEnv handles path_env correctly in various scenarios."""

    def test_path_env_explicit_override(self, unified_env_factory):
        test_path = Path(tempfile.gettempdir()) / "explicit_test_path"
        test_path.mkdir(parents=True, exist_ok=True)

        try:
            env = unified_env_factory(
                env_type="base",
                env_id=1,
                state_config_type="default",
                path_env=test_path,
            )
            assert env.path_env == test_path
        finally:
            if test_path.exists():
                shutil.rmtree(test_path, ignore_errors=True)

    def test_path_env_automatic_detection(self, unified_env_factory):
        env = unified_env_factory(
            env_type="base",
            env_id=2,
            state_config_type="default",
        )
        assert env.path_env is not None
        assert isinstance(env.path_env, Path)

    def test_path_env_fallback_warning(self, unified_env_factory, caplog):
        with caplog.at_level(logging.WARNING):
            env = unified_env_factory(
                env_type="base",
                env_id=3,
                state_config_type="default",
            )
            assert env.path_env is not None

    def test_path_env_not_none_after_init(self, unified_env_factory):
        env = unified_env_factory(
            env_type="base",
            env_id=4,
            state_config_type="default",
        )
        assert env.path_env is not None
        assert isinstance(env.path_env, Path)


class TestTransformStateLog:
    """Unit tests for BaseEnv.transform_state_log"""

    @pytest.fixture
    def env(self, unified_env_factory) -> BaseEnv:
        return unified_env_factory(sampling_time=1)

    def test_empty_state_log_raises_runtime_error(self, env: BaseEnv):
        """Should raise RuntimeError when state_log is empty."""
        env.state_log = []

        with pytest.raises(RuntimeError, match="State log is empty"):
            env.transform_state_log()

    def test_returns_dataframe_with_correct_index_live_mode(self, env: BaseEnv):
        """Should return DataFrame with proper datetime index in live mode."""
        env.episode_timer = pd.Timestamp("2024-01-01 10:00:00")
        env.state_log = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        result = env.transform_state_log()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        start_time = pd.Timestamp("2024-01-01 10:00:00")
        assert result.index[0] == start_time
        assert result.index[1] == start_time + pd.Timedelta(seconds=1)
        assert list(result.iloc[0]) == [1.0, 2.0]
        assert list(result["b"]) == [2.0, 4.0]

    def test_returns_dataframe_with_correct_index_scenario_mode(self, env: BaseEnv):
        """Should return DataFrame with proper datetime index in scenario mode."""
        start_date = pd.Timestamp("2024-01-01 10:00:00")
        env.scenario_manager = Mock()
        env.scenario_manager.scenarios = pd.DataFrame(index=[start_date])
        env._scenario_offset = 5
        env.state_log = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        result = env.transform_state_log()

        expected_start = start_date + 5 * pd.Timedelta(seconds=1)
        assert result.index[0] == expected_start
        assert len(result) == 2
        assert list(result.iloc[0]) == [1.0, 2.0]
        assert list(result["b"]) == [2.0, 4.0]

    def test_state_log_with_various_data_types(self, env: BaseEnv):
        """Should handle different numeric data types in state_log."""
        env.episode_timer = pd.Timestamp("2024-01-01")
        env.state_log = [{"a": 1, "b": 2.1, "c": True, "d": "String1"}, {"a": 3, "b": 4}]

        result = env.transform_state_log()

        assert result.shape == (2, 4)
        assert result.iloc[0, 0] == 1
        assert result.iloc[0, 1] == 2.1
        assert result.iloc[0, 2] == True  # noqa: E712
        assert result.iloc[0, 3] == "String1"

    def test_different_sim_steps_per_sample(self, env: BaseEnv):
        """Should return DataFrame with proper datetime index in live mode."""
        env.sim_steps_per_sample = 5
        env.episode_timer = pd.Timestamp("2024-01-01 10:00:00")
        env.state_log = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        result = env.transform_state_log()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        start_time = pd.Timestamp("2024-01-01 10:00:00")
        assert result.index[0] == start_time
        assert result.index[1] == start_time + pd.Timedelta(seconds=0.2)  # 1.0 / 5
        assert list(result.iloc[0]) == [1.0, 2.0]
        assert list(result["b"]) == [2.0, 4.0]


class TestExportStateLog:
    """Unit tests for BaseEnv.export_state_log."""

    @pytest.fixture
    def env(self, unified_env_factory) -> BaseEnv:
        return unified_env_factory(sampling_time=1)

    def test_export_state_log(self, env: BaseEnv, tmp_path):
        env.episode_timer = pd.Timestamp("2024-01-01 10:00:00")
        env.state_log = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        path = tmp_path / "state_log.csv"
        env.export_state_log(path)

        report = pd.read_csv(path, sep=";", index_col=0)
        assert list(report.columns) == ["a", "b"]
        assert list(report["b"]) == [2.0, 4.0]

    def test_export_filters_columns(self, env: BaseEnv, tmp_path):
        env.episode_timer = pd.Timestamp("2024-01-01 10:00:00")
        env.state_log = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

        env.export_state_log(tmp_path / "state_log.csv", names=["b"])

        report = pd.read_csv(tmp_path / "state_log.csv", sep=";", index_col=0)
        assert list(report.columns) == ["b"]
        assert list(report["b"]) == [2.0, 5.0]
