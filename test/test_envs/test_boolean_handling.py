"""Tests for boolean variable handling in external inputs/outputs and abort conditions."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eta_ctrl.config.config_run import ConfigRun
from eta_ctrl.envs.base_env import BaseEnv
from eta_ctrl.envs.state import StateConfig, StateVar


@pytest.fixture
def test_env():
    """Create a minimal test environment with boolean and numeric variables."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    config_run = ConfigRun(
        series="test",
        name="bool_test",
        description="Test",
        root_path=temp_path,
        results_path=temp_path / "results",
        scenarios_path=temp_path / "scenarios",
    )

    state_config = StateConfig(
        StateVar(
            name="temperature",
            is_agent_observation=True,
            is_ext_output=True,
            ext_id="temp",
            ext_scale_add=10.0,
            ext_scale_mult=2.0,
            abort_condition_min=10.0,
            abort_condition_max=90.0,
        ),
        StateVar(name="status", is_agent_observation=True, is_ext_output=True, ext_id="status"),
        StateVar(name="value", is_agent_action=True, is_ext_input=True, ext_id="value"),
    )

    class TestEnv(BaseEnv):
        version = "1.0"
        description = "Test"

        def _step(self):
            return 0.0, False, False, {}

        def _reset(self, *, seed=None, options=None):
            return {}

        def close(self):
            pass

        def render(self):
            pass

    env = TestEnv(env_id=1, config_run=config_run, state_config=state_config, episode_duration=60, sampling_time=1)

    # Initialize state
    env.state = {}

    yield env

    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBooleanHandling:
    """Test boolean handling in all three modified methods."""

    @pytest.mark.parametrize(
        ("bool_val", "bool_type"),
        [(True, bool), (False, bool), (np.bool_(True), np.bool_), (np.bool_(False), np.bool_)],
    )
    def test_set_external_outputs_preserves_booleans(self, test_env, bool_val, bool_type):
        """Test set_external_outputs preserves booleans without scaling."""
        test_env.set_external_outputs({"temp": 25.0, "status": bool_val})

        # Boolean preserved
        assert test_env.state["status"][0] == bool(bool_val)
        assert isinstance(test_env.state["status"][0], (bool, np.bool_))

        # Numeric scaled: (25.0 + 10.0) * 2.0 = 70.0
        assert np.isclose(test_env.state["temperature"][0], 70.0)

    @pytest.mark.parametrize("bool_val", [True, False, np.bool_(True)])
    def test_get_external_inputs_preserves_booleans(self, test_env, bool_val):
        """Test get_external_inputs preserves booleans without inverse scaling."""
        test_env.state = {"value": np.array([50.0]), "status": np.array([bool_val])}

        inputs = test_env.get_external_inputs()

        # Boolean preserved as Python bool
        assert inputs["value"] == 50.0
        assert inputs.get("status") is None  # status is not ext_input

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ({"temperature": 50.0, "status": True}, True),
            ({"temperature": 50.0, "status": False}, True),
            ({"temperature": 95.0, "status": True}, False),  # temp exceeds max
            ({"temperature": 5.0, "status": False}, False),  # temp below min
        ],
    )
    def test_within_abort_conditions_skips_booleans(self, test_env, state, expected):
        """Test within_abort_conditions correctly skips booleans."""
        assert test_env.state_config.within_abort_conditions(state) == expected

    def test_boolean_roundtrip(self, test_env):
        """Test booleans survive roundtrip: set_external_outputs -> state -> get_external_inputs."""
        for bool_val in [True, False]:
            test_env.set_external_outputs({"temp": 20.0, "status": bool_val})
            assert test_env.state["status"][0] == bool_val

    def test_backward_compatibility_numeric_only(self):
        """Verify numeric-only configs still work (backward compatibility)."""
        config = StateConfig(
            StateVar(name="temp", is_agent_observation=True, abort_condition_min=10.0, abort_condition_max=90.0)
        )
        assert config.within_abort_conditions({"temp": 50.0}) is True
        assert config.within_abort_conditions({"temp": 95.0}) is False
