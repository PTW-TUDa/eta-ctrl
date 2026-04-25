import numpy as np
import pytest

from eta_ctrl.envs.state import StateConfig, StateVar

MODEL_IMPORT = "test.resources.pyomo_basic_model.PyomoBasicModel"


class TestPyomoSimEnv:
    @pytest.fixture(scope="class")
    def env(self, unified_env_factory):
        state_config = StateConfig(
            StateVar(name="x_agent", is_agent_action=True, is_ext_input=True, low_value=-5, high_value=5, ext_id="x"),
            StateVar(name="temp0_agent", is_agent_observation=True, is_ext_output=True, ext_id="temp0"),
        )
        return unified_env_factory(env_type="pyomo", state_config=state_config)

    def test_step(self, env):
        """Test the actual simulation process"""

        # Set initial value (usually done by reset)
        env.model.pyo_update_params({"temp0": 54})

        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))

        assert obs["temp0_agent"] == 52.0

        obs, _, _, _, _ = env.step(np.array([1.0], dtype=np.float32))

        assert obs["temp0_agent"] == 54.0


class TestPyomoSimEnvFail:
    def test_missing_expression(self, unified_env_factory):
        state_config = StateConfig(
            StateVar(name="u", is_agent_action=True, low_value=-5, high_value=5),
            StateVar(name="x0", is_agent_observation=True, is_ext_output=True),
        )
        msg = "PyomoModel 'PyomoBasicModel' is not compatible with the PyomoSimEnv, see the documentation."
        with pytest.raises(NotImplementedError, match=msg) as exc_info:
            unified_env_factory(env_type="pyomo", state_config=state_config)

        assert str(exc_info.value.__context__) == "\"Missing 'x0' in start_value_mapping of 'PyomoBasicModel'\""


class TestPyomoSimEnvStringRepresentations:
    @pytest.fixture(scope="class")
    def concrete_pyomo_env(self, unified_env_factory):
        """Create a concrete PyomoSimEnv instance for testing."""
        return unified_env_factory(
            env_type="pyomo",
            env_id=42,
            episode_duration=7200,  # 2 hours
            sampling_time=300,  # 5 minutes
        )

    def test_str_representation(self, concrete_pyomo_env):
        expected = (
            "TestPyomoSimEnv(id=42, 2 actions, 2 observations, Episode 0, Step 0/24), PyomoModel: PyomoBasicModel"
        )
        assert str(concrete_pyomo_env) == expected

    def test_repr_representation(self, concrete_pyomo_env):
        expected = (
            "TestPyomoSimEnv(env_id=42, run_name='pyomo_test_run', n_episodes=0, n_steps=0, "
            "episode_duration=7200.0, sampling_time=300.0, "
            "model_import='test.resources.pyomo_basic_model.PyomoBasicModel')"
        )
        assert repr(concrete_pyomo_env) == expected
