"""Tests for StateVar class."""

import numpy as np
import pytest

from eta_ctrl.envs.state import StateVar


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

    @pytest.mark.parametrize(
        ("name"),
        [
            pytest.param("interact", id="interact"),
            pytest.param("scenario", id="scenario"),
        ],
    )
    def test_missing_params(self, name):
        msg = f"Variable {name} is either missing `{name}_id` or `from_{name}`"
        with pytest.raises(KeyError, match=msg):
            StateVar(name=name, **{f"from_{name}": True})

    @pytest.fixture(scope="class")
    def state_var_scenario(self):
        return StateVar(
            name="foo",
            is_agent_action=True,
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
        assert state_var_scenario.is_agent_action is True
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
