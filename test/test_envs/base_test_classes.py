"""Shared base test classes and fixtures for environment string representation tests."""

import pathlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eta_ctrl.config import ConfigRun
from eta_ctrl.envs import BaseEnv, LiveEnv, PyomoSimEnv, SimEnv, StateConfig, StateVar
from eta_ctrl.timeseries.scenario_manager import ScenarioManager


class _ScenarioManagerStub:
    """Minimal scenario manager stub for method-level tests."""

    def __init__(self, value: float = 2.0, offset: int = 2) -> None:
        self.value = value
        self.offset = offset

    def compute_episode_offset(self, _rng) -> int:
        return self.offset

    def get_scenario_state_var(self, n_step: int, state_var: StateVar):
        return np.array([self.value + n_step + state_var.ext_scale_add])


class DummyScenarioManager(ScenarioManager):
    """Dummy class for testing purposes"""

    def __init__(self) -> None:
        self.scenarios = pd.DataFrame()

    def get_scenario_state(self, n_steps):
        return {}

    def get_scenario_state_with_duration(self, n_step, duration):
        return {}

    def _get_data(self, n_step, duration=1, names=None):
        return {}


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
            root_path=temp_path,
            results_path=temp_path / "results" if create_subdirs else temp_path,
            scenarios_path=temp_path / "scenarios" if create_subdirs else temp_path,
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
                StateVar(name="test_action", is_agent_action=True, low_value=0, high_value=1),
                StateVar(name="test_obs", is_agent_observation=True),
            ),
            "basic": lambda: StateConfig(
                StateVar(name="test_action", is_agent_action=True, low_value=0, high_value=100),
                StateVar(name="test_obs", is_agent_observation=True, low_value=0, high_value=100),
            ),
            "scenario": lambda: StateConfig(
                StateVar(name="scen1", from_scenario=True),
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
            "method_test": lambda: StateConfig(
                StateVar(name="act", is_agent_action=True, low_value=0.0, high_value=10.0),
                StateVar(name="obs", is_agent_observation=True, low_value=-10.0, high_value=100.0),
                StateVar(
                    name="ext_in_state", is_ext_input=True, ext_id="ext.in", ext_scale_add=10.0, ext_scale_mult=2.0
                ),
                StateVar(
                    name="ext_out_state", is_ext_output=True, ext_id="ext.out", ext_scale_add=1.0, ext_scale_mult=3.0
                ),
                StateVar(name="scen", from_scenario=True, ext_scale_add=2.0, ext_scale_mult=4.0),
            ),
            "validation": lambda: StateConfig(
                StateVar(name="a1", is_agent_action=True, low_value=-1.0, high_value=2.0),
                StateVar(name="a2", is_agent_action=True, low_value=0.0, high_value=5.0),
                StateVar(name="a3", is_agent_action=True, low_value=-10.0, high_value=10.0),
                StateVar(name="obs", is_agent_observation=True, low_value=0.0, high_value=100.0),
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
    Unified factory fixture for creating any type of environment (BaseEnv, PyomoSimEnv, SimEnv, LiveEnv).
    """

    def _create_environment(
        env_type="base",
        env_id=42,
        config_run_params=None,
        state_config_type="default",
        episode_duration=7200,
        sampling_time=300,
        path_env=None,
        **env_specific_params,
    ):
        # Use default config_run_params if not provided
        if config_run_params is None:
            config_run_params = {
                "series": "test_series",
                "name": f"{env_type}_test_run",
                "description": f"Test run for {env_type} environment",
            }

        config_run, _temp_path = config_run_factory(**config_run_params)
        state_config = state_config_factory(state_config_type)

        # Common environment parameters
        common_params = {
            "env_id": env_id,
            "config_run": config_run,
            "state_config": state_config,
            "episode_duration": episode_duration,
            "sampling_time": sampling_time,
            "path_env": path_env,
        }
        # Common params can be overridden by env_specific_params (i.e. kwargs)
        all_params = {**common_params, **env_specific_params}

        if env_type == "base":
            return TestBaseEnv(**all_params)
        if env_type == "pyomo":
            # Set PyomoSimEnv specific parameters with defaults
            all_params.setdefault("model_parameters", {})
            all_params.setdefault("scenario_manager", DummyScenarioManager())
            return TestPyomoSimEnv(**all_params)
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
        if env_type == "method":
            env = _MethodTestEnv(**{**common_params, **env_specific_params})
            env.scenario_manager = _ScenarioManagerStub()
            env._scenario_rng = np.random.default_rng(0)
            return env
        error_msg = f"Unknown env_type: {env_type}. Supported types: base, pyomo, sim, live, method"
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

    def close(self):
        pass

    def render(self):
        pass


class TestPyomoSimEnv(PyomoSimEnv):
    """Concrete implementation of PyomoSimEnv for testing."""

    @property
    def model_import(self):
        return "test.resources.pyomo_basic_model.PyomoBasicModel"

    @property
    def version(self):
        return "v1.0.0"

    @property
    def description(self):
        return "Test PyomoSimEnv for string representation testing"

    def _build_model(self):
        pass

    def _solve_model(self):
        return {}

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

    def render(self):
        pass


class _MethodTestEnv(BaseEnv):
    """BaseEnv subclass with non-trivial _step/_reset for method-level tests."""

    @property
    def version(self):
        return "v1.0.0"

    @property
    def description(self):
        return "BaseEnv method test env"

    def _step(self):
        self.state["obs"] = np.array([float(self.state["act"].item())])
        return 1.0, False, False, {"source": "_step"}

    def _reset(self, *, seed=None, options=None):
        self.state["obs"] = np.array([5.0])
        return {"reset": True}

    def close(self):
        pass

    def render(self):
        pass
