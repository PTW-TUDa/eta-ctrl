"""Tests for StateConfig class."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict

from eta_ctrl.envs.state import StateConfig, StateVar


class TestStateVar:
    @pytest.fixture(scope="class")
    def state_var_default(self):
        return StateVar(name="foo")

    @pytest.fixture(scope="class")
    def state_var_ext(self):
        return StateVar("foo", is_ext_input=True)

    def test_defaults(self, state_var_default: StateVar):
        assert state_var_default["name"] == "foo"
        assert state_var_default.is_agent_action is False
        assert state_var_default.is_agent_observation is False
        assert state_var_default.is_ext_input is False
        assert state_var_default.is_ext_output is False
        assert state_var_default.low_value == -np.inf
        assert state_var_default.high_value == np.inf
        assert state_var_default.ext_id is None
        assert state_var_default.abort_condition_min == -np.inf
        assert state_var_default.abort_condition_max == np.inf

    @pytest.fixture(scope="class")
    def state_var_scenario(self):
        return StateVar(
            name="foo",
            is_agent_observation=True,
            low_value=0,
            high_value=1,
            from_scenario=True,
            is_ext_input=True,
            scenario_id="0",
            scenario_scale_add=1,
            scenario_scale_mult=2,
        )

    def test_scenario_var(self, state_var_scenario):
        assert state_var_scenario.name == "foo"
        assert state_var_scenario.is_agent_observation is True
        assert state_var_scenario.low_value == 0
        assert state_var_scenario.high_value == 1
        assert state_var_scenario.from_scenario is True
        assert state_var_scenario.is_ext_input is True
        assert state_var_scenario.scenario_id == "0"
        assert state_var_scenario.scenario_scale_add == 1.0
        assert state_var_scenario.scenario_scale_mult == 2.0

    def test_state_var_ext_id_should_be_name_by_default(self, state_var_scenario):
        assert state_var_scenario.ext_id == state_var_scenario.name

    @pytest.fixture(scope="class")
    def state_var_interact(self):
        return StateVar(
            name="foo",
            is_agent_action=True,
            low_value=0,
            high_value=1,
            from_interact=True,
            interact_id=0,
            interact_scale_add=1,
            interact_scale_mult=2,
        )

    def test_interact_var(self, state_var_interact):
        assert state_var_interact.name == "foo"
        assert state_var_interact.is_agent_action is True
        assert state_var_interact.low_value == 0
        assert state_var_interact.high_value == 1
        assert state_var_interact.from_interact is True
        assert state_var_interact.interact_id == 0
        assert state_var_interact.interact_scale_add == 1.0
        assert state_var_interact.interact_scale_mult == 2.0

    def test_from_dict(self):
        dikt = {"name": "foo", "is_agent_action": True}
        state_var = StateVar.from_dict(dikt)
        assert state_var.name == "foo"
        assert state_var.is_agent_action is True

    def test_str_representation_action_only(self):
        """Test __str__ method for action-only StateVar."""
        state_var = StateVar(name="power_output", is_agent_action=True, low_value=0, high_value=1000)
        expected = "StateVar 'power_output' (action) [0.0, 1000.0]"
        assert str(state_var) == expected

    def test_str_representation_observation_only(self):
        """Test __str__ method for observation-only StateVar."""
        state_var = StateVar(name="temperature", is_agent_observation=True, low_value=-10, high_value=50)
        expected = "StateVar 'temperature' (observation) [-10.0, 50.0]"
        assert str(state_var) == expected

    def test_str_representation_both_action_and_observation(self):
        """Test __str__ method for StateVar that is both action and observation."""
        state_var = StateVar(
            name="valve_position", is_agent_action=True, is_agent_observation=True, low_value=0, high_value=100
        )
        expected = "StateVar 'valve_position' (action/observation) [0.0, 100.0]"
        assert str(state_var) == expected

    def test_str_representation_neither_action_nor_observation(self):
        """Test __str__ method for StateVar that is neither action nor observation."""
        state_var = StateVar(name="internal_state")
        expected = "StateVar 'internal_state' (variable)"
        assert str(state_var) == expected

    def test_str_representation_infinite_range(self):
        """Test __str__ method for StateVar with infinite range."""
        state_var = StateVar(name="unlimited_var", is_agent_action=True)
        expected = "StateVar 'unlimited_var' (action)"
        assert str(state_var) == expected

    def test_str_representation_partial_infinite_range(self):
        """Test __str__ method for StateVar with partially infinite range."""
        state_var = StateVar(name="min_limited", is_agent_action=True, low_value=0)
        expected = "StateVar 'min_limited' (action) [0.0, inf]"
        assert str(state_var) == expected

    def test_repr_representation_minimal(self):
        """Test __repr__ method for minimal StateVar."""
        state_var = StateVar(name="simple_var")
        expected = "StateVar(name='simple_var')"
        assert repr(state_var) == expected

    def test_repr_representation_full_action(self):
        """Test __repr__ method for full action StateVar."""
        state_var = StateVar(name="heating_power", is_agent_action=True, low_value=0, high_value=5000)
        expected = "StateVar(name='heating_power', is_agent_action=True, low_value=0.0, high_value=5000.0)"
        assert repr(state_var) == expected

    def test_repr_representation_observation_with_infinite_high(self):
        """Test __repr__ method for observation with infinite high value."""
        state_var = StateVar(name="error_signal", is_agent_observation=True, low_value=-100)
        expected = "StateVar(name='error_signal', is_agent_observation=True, low_value=-100.0)"
        assert repr(state_var) == expected

    def test_repr_representation_both_action_observation(self):
        """Test __repr__ method for StateVar that is both action and observation."""
        state_var = StateVar(
            name="feedback_var", is_agent_action=True, is_agent_observation=True, low_value=10, high_value=90
        )
        expected = (
            "StateVar(name='feedback_var', is_agent_action=True, "
            "is_agent_observation=True, low_value=10.0, high_value=90.0)"
        )
        assert repr(state_var) == expected


def create_box(low: list[float], high: list[float]):
    return Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32))


class TestStateConfig:
    @pytest.fixture(scope="class")
    def state_config(self):
        return StateConfig(
            StateVar(
                name="action1",
                is_agent_action=True,
                low_value=0,
                high_value=1,
                abort_condition_min=0,
                abort_condition_max=0.5,
            ),
            StateVar(name="observation1", is_agent_observation=True, low_value=2, high_value=3),
            StateVar(
                name="action2",
                is_agent_action=True,
                low_value=4,
                high_value=5,
                abort_condition_min=4,
                abort_condition_max=4.5,
            ),
            StateVar(name="observation2", is_agent_observation=True, low_value=6, high_value=7),
        )

    @pytest.fixture(scope="class")
    def state_config_nan(self):
        return StateConfig(
            StateVar(name="action1", is_agent_action=True),
            StateVar(name="observation1", is_agent_observation=True),
        )

    @pytest.fixture(scope="class")
    def config_from_test_env_file(self, config_resources_path):
        return StateConfig.from_file(path=config_resources_path, filename="test_env_state_config.toml")

    @pytest.fixture(scope="class")
    def config_from_test_env_csv_file(self, config_resources_path):
        return StateConfig.from_file(path=config_resources_path, filename="test_env_state_config.csv")

    def test_continuous_action_space_should_include_all_and_only_agent_actions(self, state_config):
        # also tests: continuous_action_space_should_span_from_low_to_high_value_for_every_statevar
        action_space = state_config.continuous_action_space()
        assert action_space == create_box(low=[0, 4], high=[1, 5])

    def test_continuous_action_space_should_use_infinity_if_no_low_and_high_values_are_provided(self, state_config_nan):
        action_space = state_config_nan.continuous_action_space()
        assert action_space == create_box(low=[-np.inf], high=[np.inf])

    def test_continuous_observation_space_should_include_all_and_only_agent_observations(self, state_config):
        # also tests: continuous_observation_space_should_span_from_low_to_high_value_for_every_statevar
        obs_space = state_config.continuous_observation_space()
        assert obs_space == Dict(
            {
                "observation1": create_box(low=[2], high=[3]),
                "observation2": create_box(low=[6], high=[7]),
            }
        )

    def test_continuous_observation_space_should_use_infinity_if_no_low_and_high_values_are_provided(
        self, state_config_nan
    ):
        obs_space = state_config_nan.continuous_observation_space()
        assert obs_space == Dict({"observation1": create_box(low=[-np.inf], high=[np.inf])})

    def test_from_dict(self):
        state_vars = [
            {"name": "action1", "is_agent_action": True, "ext_id": "foo"},
            {"name": "action2", "is_agent_observation": True, "ext_id": "foo"},
        ]
        StateConfig.from_dict(state_vars)

    def test_from_dict_with_dataframe(self):
        import pandas as pd

        state_vars = pd.DataFrame([{"name": "foo", "is_agent_action": True}])
        stateconfig = StateConfig.from_dict(state_vars)
        assert stateconfig.actions == ["foo"]

    @pytest.mark.parametrize(
        ("vals", "truth"),
        [
            pytest.param({"action1": 0, "action2": 4}, True, id="within"),
            pytest.param({"action1": -1, "action2": 5}, False, id="outside"),
        ],
    )
    def test_abort_conditions(self, state_config: StateConfig, vals, truth: bool):
        assert state_config.within_abort_conditions(vals) is truth

    def test_map_ids(self):
        config = StateConfig(
            StateVar(name="foo1", is_ext_input=True, ext_id="bar1"),
            StateVar(name="foo2", is_ext_output=True, ext_id="bar2"),
        )
        for name, ext_id in config.map_ext_ids.items():
            assert name == config.rev_ext_ids[ext_id]

    def test_get_items(self, state_config: StateConfig):
        for name in state_config.actions + state_config.observations:
            assert state_config.loc[name] == state_config.vars[name]

    def test_from_toml_file(self, config_from_test_env_file):
        assert config_from_test_env_file is not None

    def test_from_csv_file(self, config_from_test_env_csv_file):
        assert config_from_test_env_csv_file is not None

    def test_from_file_respects_suffix(self, config_from_test_env_csv_file, config_from_test_env_file):
        """Loading from CSV should include `additional_param`, TOML should not."""
        assert "additional_param" in config_from_test_env_csv_file.observations
        assert "additional_param" not in config_from_test_env_file.observations

    def test_state_params(self):
        state_params = {"extra_param": True}
        statevars = [
            {"name": "action1", "is_agent_action": "extra_param"},
        ]
        state_config = StateConfig.from_dict(statevars, state_params=state_params)
        assert state_config.loc["action1"]["is_agent_action"] is True

    def test_state_params_fail(self):
        statevars = [
            {"name": "action1", "is_agent_action": "extra_param"},
        ]
        error_msg = "Parameter extra_param needs to be specified in state_params"
        with pytest.raises(KeyError, match=error_msg):
            StateConfig.from_dict(statevars)

    def test_str_representation_standard_config(self, state_config: StateConfig):
        """Test __str__ method for standard StateConfig with actions and observations."""
        expected = "StateConfig with 2 actions, 2 observations (4 total variables)"
        assert str(state_config) == expected

    def test_str_representation_empty_config(self):
        """Test __str__ method for empty StateConfig."""
        empty_config = StateConfig()
        expected = "StateConfig with 0 actions, 0 observations (0 total variables)"
        assert str(empty_config) == expected

    def test_str_representation_actions_only(self):
        """Test __str__ method for StateConfig with only actions."""
        config = StateConfig(
            StateVar(name="action1", is_agent_action=True),
            StateVar(name="action2", is_agent_action=True),
        )
        expected = "StateConfig with 2 actions, 0 observations (2 total variables)"
        assert str(config) == expected

    def test_str_representation_observations_only(self):
        """Test __str__ method for StateConfig with only observations."""
        config = StateConfig(
            StateVar(name="obs1", is_agent_observation=True),
            StateVar(name="obs2", is_agent_observation=True),
            StateVar(name="obs3", is_agent_observation=True),
        )
        expected = "StateConfig with 0 actions, 3 observations (3 total variables)"
        assert str(config) == expected

    def test_str_representation_mixed_with_other_vars(self):
        """Test __str__ method for StateConfig with actions, observations, and other variables."""
        config = StateConfig(
            StateVar(name="action1", is_agent_action=True),
            StateVar(name="obs1", is_agent_observation=True),
            StateVar(name="internal_var1"),  # Neither action nor observation
            StateVar(name="internal_var2"),  # Neither action nor observation
        )
        expected = "StateConfig with 1 actions, 1 observations (4 total variables)"
        assert str(config) == expected

    def test_repr_representation_standard_config(self, state_config: StateConfig):
        """Test __repr__ method for standard StateConfig."""
        repr_result = repr(state_config)
        assert repr_result.startswith("StateConfig(")
        assert repr_result.endswith(")")
        assert "actions=['action1', 'action2']" in repr_result
        # Check that observations are present (order may vary)
        assert "observations=" in repr_result
        assert "'observation1'" in repr_result
        assert "'observation2'" in repr_result

    def test_repr_representation_empty_config(self):
        """Test __repr__ method for empty StateConfig."""
        empty_config = StateConfig()
        expected = "StateConfig(actions=[], observations=[])"
        assert repr(empty_config) == expected

    def test_repr_representation_large_config_with_truncation(self):
        """Test __repr__ method for large StateConfig with truncation."""
        actions = [StateVar(name=f"action_{i}", is_agent_action=True) for i in range(5)]
        observations = [StateVar(name=f"obs_{i}", is_agent_observation=True) for i in range(6)]
        config = StateConfig(*actions, *observations)

        # Should show first 3 of each with ... indicating truncation
        repr_str = repr(config)
        assert "actions=['action_0', 'action_1', 'action_2', ...]" in repr_str
        assert "observations=" in repr_str
        assert "..." in repr_str  # Should have truncation indicator

    def test_repr_representation_exactly_three_items(self):
        """Test __repr__ method for StateConfig with exactly 3 actions and observations (no truncation)."""
        actions = [StateVar(name=f"action_{i}", is_agent_action=True) for i in range(3)]
        observations = [StateVar(name=f"obs_{i}", is_agent_observation=True) for i in range(3)]
        config = StateConfig(*actions, *observations)

        repr_str = repr(config)
        # Should not have ... since we have exactly 3 items
        assert "actions=['action_0', 'action_1', 'action_2']" in repr_str
        assert "..." not in repr_str

    def test_from_file_config_str_repr(self, config_from_test_env_file):
        """Test __str__ and __repr__ methods using config loaded from file."""
        # Based on the TOML file: 1 action (torque), 4 observations (th, cos_th, sin_th, th_dot)
        str_result = str(config_from_test_env_file)
        assert "1 actions" in str_result
        assert "4 observations" in str_result
        assert "5 total variables" in str_result

        repr_result = repr(config_from_test_env_file)
        assert "actions=['torque']" in repr_result
        assert "observations=" in repr_result
        # With 4 observations, it should be truncated (showing first 3 with ...)
        assert "..." in repr_result
        # Check that at least some observation names are present (first 3)
        assert "observations=[" in repr_result

    def test_manual_creation_no_source_file(self):
        """Test that manually created StateConfig has no source path."""
        config = StateConfig(
            StateVar(name="test_action", is_agent_action=True), StateVar(name="test_obs", is_agent_observation=True)
        )
        assert config.source_file is None

    def test_from_dict_no_source_file(self):
        """Test that from_dict without source path sets source_file to None."""
        state_vars = [{"name": "action1", "is_agent_action": True}, {"name": "obs1", "is_agent_observation": True}]
        config = StateConfig.from_dict(state_vars)
        assert config.source_file is None

    def test_from_file_sets_source_file(self, config_resources_path, config_from_test_env_file):
        """Test that from_file sets source_file to the file path."""
        path = config_resources_path / "test_env_state_config.toml"
        assert config_from_test_env_file.source_file == path

    def test_str_representation_withoutsource_file(self):
        """Test __str__ method when no source path is available."""
        config = StateConfig(
            StateVar(name="action1", is_agent_action=True), StateVar(name="obs1", is_agent_observation=True)
        )
        str_result = str(config)
        expected = "StateConfig with 1 actions, 1 observations (2 total variables)"
        assert str_result == expected
        assert "from" not in str_result

    def test_str_representation_with_file_source_file(self, config_from_test_env_file):
        """Test __str__ method with real file path from from_file."""
        str_result = str(config_from_test_env_file)
        assert "StateConfig with 1 actions, 4 observations (5 total variables) from" in str_result
        assert str(config_from_test_env_file.source_file) in str_result

    def test_repr_representation_unchanged_with_source_file(self, config_from_test_env_file):
        """Test that __repr__ method doesn't include source path (developer format)."""
        repr_result = repr(config_from_test_env_file)
        expected = "StateConfig(actions=['torque'], observations=['cos_th', 'sin_th', 'th', ...])"
        assert repr_result == expected
        # Ensure path is not in repr (it's for developers, not end users)
        assert str(config_from_test_env_file.source_file) not in repr_result
        assert "from" not in repr_result

    def test_from_file_loads_from_given_path(self, tmp_path: Path):
        config_file = tmp_path / "my_state_config.toml"
        config_file.write_text("[[actions]]\nname = 'u'\n", encoding="utf-8")

        config = StateConfig.from_file(path=tmp_path, filename="my_state_config.toml")

        assert config.actions == ["u"]
        assert config.source_file == config_file

    def test_from_file_falls_back_to_environments_subdir(self, tmp_path: Path):
        config_file = tmp_path / "environments" / "my_state_config.toml"
        config_file.parent.mkdir()
        config_file.write_text("[[actions]]\nname = 'u'\n", encoding="utf-8")

        config = StateConfig.from_file(path=tmp_path, filename="my_state_config.toml")
        assert config.actions == ["u"]
        assert config.source_file == config_file

    def test_from_file_raises_if_file_not_found(self, tmp_path: Path):
        expected_direct = tmp_path / "missing.toml"
        expected_fallback = tmp_path / "environments" / "missing.toml"
        error_msg = f"StateConfig file not found at {expected_direct} or {expected_fallback}".replace("\\", "\\\\")

        with pytest.raises(FileNotFoundError, match=error_msg):
            StateConfig.from_file(path=tmp_path, filename="missing.toml")
