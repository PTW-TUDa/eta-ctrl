"""Shared base test classes and fixtures for environment string representation tests."""

import pathlib
import tempfile
from datetime import datetime
from pathlib import Path

import pyomo.environ as pyo
import pytest

from eta_ctrl.config.config_run import ConfigRun
from eta_ctrl.envs.base_env import BaseEnv
from eta_ctrl.envs.live_env import LiveEnv
from eta_ctrl.envs.pyomo_env import PyomoEnv
from eta_ctrl.envs.sim_env import SimEnv
from eta_ctrl.envs.state import StateConfig, StateVar


@pytest.fixture(scope="class")
def temp_directory_factory():
    """Factory fixture for creating temporary directories with automatic cleanup."""
    directories = []

    def _create_temp_directory():
        temp_dir = tempfile.mkdtemp()
        directories.append(temp_dir)
        return Path(temp_dir)

    yield _create_temp_directory

    # Cleanup all created directories
    for directory in directories:
        import shutil

        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="class")
def config_run_factory(temp_directory_factory):
    """Factory fixture for creating ConfigRun instances."""

    def _create_config_run(series="test_series", name="test_run", description="Test run", create_subdirs=True):
        temp_path = temp_directory_factory()
        config = ConfigRun(
            series=series,
            name=name,
            description=description,
            path_root=temp_path,
            path_results=temp_path / "results" if create_subdirs else temp_path,
            path_scenarios=temp_path / "scenarios" if create_subdirs else temp_path,
        )
        return config, temp_path

    return _create_config_run


@pytest.fixture(scope="class")
def state_config_factory():
    """Factory fixture for creating StateConfig instances of different types."""

    def _create_state_config(config_type="default"):
        config_map = {
            "default": lambda: StateConfig(
                StateVar(name="heating_power", is_agent_action=True, low_value=0, high_value=5000),
                StateVar(name="cooling_power", is_agent_action=True, low_value=0, high_value=3000),
                StateVar(name="room_temp", is_agent_observation=True, low_value=15, high_value=30),
                StateVar(name="outside_temp", is_agent_observation=True, low_value=-20, high_value=40),
            ),
            "many_actions": lambda: StateConfig(
                *[StateVar(name=f"action_{i}", is_agent_action=True, low_value=0, high_value=100) for i in range(8)],
                StateVar(name="single_obs", is_agent_observation=True, low_value=0, high_value=1),
            ),
            "minimal": lambda: StateConfig(
                StateVar(name="test_action", is_agent_action=True),
                StateVar(name="test_obs", is_agent_observation=True),
            ),
            "basic": lambda: StateConfig(
                StateVar(name="test_action", is_agent_action=True, low_value=0, high_value=100),
                StateVar(name="test_obs", is_agent_observation=True, low_value=0, high_value=100),
            ),
            "multi_action": lambda: StateConfig(
                StateVar(name="heating_power", is_agent_action=True, low_value=0, high_value=5000),
                StateVar(name="cooling_power", is_agent_action=True, low_value=0, high_value=3000),
                StateVar(name="room_temp", is_agent_observation=True, low_value=15, high_value=30),
                StateVar(name="outside_temp", is_agent_observation=True, low_value=-20, high_value=40),
            ),
            "sim": lambda: StateConfig(
                StateVar(name="valve_control", is_agent_action=True, low_value=0, high_value=100),
                StateVar(name="temperature_reading", is_agent_observation=True, low_value=0, high_value=80),
                StateVar(name="pressure_reading", is_agent_observation=True, low_value=0, high_value=10),
            ),
            "live": lambda: StateConfig(
                StateVar(name="setpoint_command", is_agent_action=True, low_value=0, high_value=100),
                StateVar(name="actual_value", is_agent_observation=True, low_value=0, high_value=100),
                StateVar(name="error_signal", is_agent_observation=True, low_value=-50, high_value=50),
            ),
        }

        if config_type not in config_map:
            error_msg = f"Unknown config_type: {config_type}"
            raise ValueError(error_msg)

        return config_map[config_type]()

    return _create_state_config


@pytest.fixture(scope="class")
def unified_env_factory(config_run_factory, state_config_factory):
    """
    Unified factory fixture for creating any type of environment (BaseEnv, PyomoEnv, SimEnv, LiveEnv).
    """

    def _create_environment(
        env_type="base",
        env_id=42,
        config_run_params=None,
        state_config_type="default",
        scenario_time_begin=datetime(2023, 6, 15, 8, 0),
        scenario_time_end=datetime(2023, 6, 15, 20, 0),
        episode_duration=7200,
        sampling_time=300,
        **env_specific_params,
    ):
        # Use default config_run_params if not provided
        if config_run_params is None:
            config_run_params = {
                "series": "test_series",
                "name": f"{env_type}_test_run",
                "description": f"Test run for {env_type} environment",
            }

        config_run, temp_path = config_run_factory(**config_run_params)
        state_config = state_config_factory(state_config_type)

        # Common environment parameters
        common_params = {
            "env_id": env_id,
            "config_run": config_run,
            "state_config": state_config,
            "scenario_time_begin": scenario_time_begin,
            "scenario_time_end": scenario_time_end,
            "episode_duration": episode_duration,
            "sampling_time": sampling_time,
        }

        if env_type == "base":
            return TestBaseEnv(**common_params)
        if env_type == "pyomo":
            # Extract PyomoEnv specific parameters with defaults
            model_parameters = env_specific_params.get("model_parameters", {})
            prediction_horizon = env_specific_params.get("prediction_horizon", 3600.0)
            n_prediction_steps = env_specific_params.get("n_prediction_steps", 12)
            return TestPyomoEnv(
                **common_params,
                model_parameters=model_parameters,
                prediction_horizon=prediction_horizon,
                n_prediction_steps=n_prediction_steps,
            )
        if env_type == "sim":
            # Extract SimEnv specific parameters with defaults
            fmu_name = env_specific_params.get("fmu_name", "test_model.fmu")
            sim_steps_per_sample = env_specific_params.get("sim_steps_per_sample", 10)
            return TestSimEnv(
                **common_params,
                fmu_name=fmu_name,
                sim_steps_per_sample=sim_steps_per_sample,
            )
        if env_type == "live":
            # Extract LiveEnv specific parameters with defaults
            config_name = env_specific_params.get("config_name", "test_config")
            max_errors = env_specific_params.get("max_errors", 25)
            return TestLiveEnv(
                **common_params,
                config_name=config_name,
                max_errors=max_errors,
            )
        error_msg = f"Unknown env_type: {env_type}. Supported types: base, pyomo, sim, live"
        raise ValueError(error_msg)

    return _create_environment


class TestBaseEnv(BaseEnv):
    """Concrete implementation of BaseEnv for testing."""

    @property
    def version(self):
        return "v2.1.0"

    @property
    def description(self):
        return "Test BaseEnv for string representation testing"

    def _step(self):
        """Implement abstract _step method."""
        return 0.0, False, False, {}

    def _reset(self, *, seed=None, options=None):
        """Implement abstract _reset method."""
        return {}

    def step(self, action):
        self.n_steps += 1
        self.n_steps_longtime += 1
        return {}, 0.0, False, False, {}

    def close(self):
        pass

    def render(self):
        pass


class TestPyomoEnv(PyomoEnv):
    """Concrete implementation of PyomoEnv for testing."""

    @property
    def version(self):
        return "v1.0.0"

    @property
    def description(self):
        return "Test PyomoEnv for string representation testing"

    def _model(self):
        return pyo.AbstractModel()

    def _build_model(self):
        pass

    def _solve_model(self):
        return {}

    def _step(self):
        """Implement abstract _step method."""
        return 0.0, False, False, {}

    def _reset(self, *, seed=None, options=None):
        """Implement abstract _reset method."""
        return {}

    def step(self, action):
        self.n_steps += 1
        self.n_steps_longtime += 1
        return {}, 0.0, False, False, {}

    def close(self):
        pass

    def render(self):
        pass


class TestSimEnv(SimEnv):
    """Concrete implementation of SimEnv for testing."""

    @property
    def version(self):
        return "v1.0.0"

    @property
    def description(self):
        return "Test SimEnv for string representation testing"

    @property
    def fmu_name(self):
        return self._fmu_name

    def __init__(self, *args, **kwargs):
        # Extract fmu_name before calling super
        self._fmu_name = kwargs.pop("fmu_name", "test_model.fmu")
        super().__init__(*args, **kwargs)
        # Override path_env for testing
        self.path_env = pathlib.Path(tempfile.gettempdir())

    def _step(self):
        """Implement abstract _step method."""
        return 0.0, False, False, {}

    def _reset(self, *, seed=None, options=None):
        """Implement abstract _reset method."""
        return {}

    def step(self, action):
        self.n_steps += 1
        self.n_steps_longtime += 1
        return {}, 0.0, False, False, {}

    def close(self):
        pass

    def render(self):
        pass


class TestLiveEnv(LiveEnv):
    """Concrete implementation of LiveEnv for testing."""

    @property
    def version(self):
        return "v1.0.0"

    @property
    def description(self):
        return "Test LiveEnv for string representation testing"

    @property
    def config_name(self):
        return self._config_name

    def __init__(self, *args, **kwargs):
        # Extract config_name before calling super
        self._config_name = kwargs.pop("config_name", "test_config")
        super().__init__(*args, **kwargs)

    def _step(self):
        """Implement abstract _step method."""
        return 0.0, False, False, {}

    def _reset(self, *, seed=None, options=None):
        """Implement abstract _reset method."""
        return {}

    def step(self, action):
        """Minimal step implementation for testing."""
        self.n_steps += 1
        self.n_steps_longtime += 1
        return {}, 0.0, False, False, {}

    def close(self):
        pass

    def render(self):
        pass
