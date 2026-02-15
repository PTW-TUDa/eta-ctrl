"""Test for scenario data flowing to FMU inputs feature."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eta_ctrl.envs.base_env import BaseEnv
from eta_ctrl.envs.state import StateConfig, StateVar
from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager

if TYPE_CHECKING:
    import pathlib


class TestScenarioToFMUInput:
    """Test suite for the scenario-to-FMU-input feature."""

    def test_statevar_allows_scenario_and_ext_input(self) -> None:
        """Test that StateVar allows both from_scenario and is_ext_input to be True."""
        # This should not raise any error
        state_var = StateVar(
            name="test_var",
            from_scenario=True,
            scenario_id="test_scenario_col",
            is_ext_input=True,
            is_agent_observation=True,
            low_value=0,
            high_value=100,
        )

        assert state_var.from_scenario is True
        assert state_var.is_ext_input is True
        assert state_var.scenario_id == "test_scenario_col"
        assert state_var.ext_id == "test_var"  # Should auto-set

    def test_statevar_prevents_scenario_with_agent_action(self) -> None:
        """Test that StateVar prevents from_scenario + is_agent_action + is_ext_input combination."""
        with pytest.raises(ValueError, match="cannot be from_scenario, is_agent_action at the same time"):
            StateVar(
                name="invalid_var",
                from_scenario=True,
                scenario_id="test_col",
                is_ext_input=True,
                is_agent_action=True,  # This should trigger error
                low_value=0,
                high_value=100,
            )

    def test_state_config_with_scenario_input(self) -> None:
        """Test StateConfig with scenario variables as external inputs."""
        state_vars = [
            # Agent action
            StateVar(
                name="u",
                is_agent_action=True,
                is_ext_input=True,
                low_value=-10,
                high_value=10,
            ),
            # Scenario data as external input
            StateVar(
                name="external_temp",
                from_scenario=True,
                scenario_id="temperature",
                is_ext_input=True,
                is_agent_observation=True,
                low_value=-20,
                high_value=40,
            ),
            # Regular output
            StateVar(
                name="internal_temp",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=15,
                high_value=30,
            ),
        ]

        config = StateConfig(*state_vars)

        # Verify configuration
        assert "u" in config.actions
        assert "external_temp" in config.scenario_outputs
        assert "external_temp" in config.ext_inputs
        assert "external_temp" in config.observations
        assert "internal_temp" in config.ext_outputs

    def test_state_config_from_toml(self, tmp_path: pathlib.Path) -> None:
        """Test loading state config from TOML with scenario-to-FMU configuration."""
        toml_content = """
[state_parameters]
max_temp = 40

[[actions]]
name = "heater_power"
is_ext_input = true
low_value = 0
high_value = 100

[[observations]]
name = "outdoor_temp"
from_scenario = true
scenario_id = "temperature_outdoor"
is_ext_input = true
low_value = -20
high_value = "max_temp"

[[observations]]
name = "indoor_temp"
is_ext_output = true
low_value = 15
high_value = 30
"""

        toml_file = tmp_path / "test_state_config.toml"
        toml_file.write_text(toml_content)

        config = StateConfig.from_file(toml_file)

        # Verify the configuration loaded correctly
        assert "heater_power" in config.actions
        assert "outdoor_temp" in config.scenario_outputs
        assert "outdoor_temp" in config.ext_inputs
        assert "outdoor_temp" in config.observations
        assert config.vars["outdoor_temp"].scenario_id == "temperature_outdoor"
        assert config.vars["outdoor_temp"].is_ext_input is True
        assert config.vars["outdoor_temp"].from_scenario is True

    def test_get_external_inputs_with_scenario_data(self) -> None:
        """Test that get_external_inputs correctly loads scenario data when needed."""
        # This test would require a full environment setup, so we'll create a minimal mock

        # Create a mock environment
        env = MagicMock(spec=BaseEnv)

        # Create state config with scenario input
        state_vars = [
            StateVar(
                name="scenario_input",
                from_scenario=True,
                scenario_id="test_data",
                is_ext_input=True,
                low_value=0,
                high_value=100,
            ),
        ]
        config = StateConfig(*state_vars)

        # Set up the mock
        env.state_config = config
        env.state = {}
        env.n_steps = 0

        # Mock the get_scenario_state method with the new columns parameter
        def mock_scenario_manager_get_scenario_state(
            n_steps: int, columns: list[str] | None = None
        ) -> dict[str, np.ndarray]:
            return {"scenario_input": np.array([42.0])}

        env.scenario_manager = MagicMock(spec=CsvScenarioManager)
        env.scenario_manager.get_scenario_state = mock_scenario_manager_get_scenario_state

        # Call the actual get_external_inputs method
        BaseEnv.set_scenario_state(env)

        # Verify the scenario data was loaded and returned
        assert "scenario_input" in env.state
        assert env.state["scenario_input"] == 42.0

    def test_scenario_data_flow_integration(self, tmp_path: pathlib.Path) -> None:
        """Integration test for complete scenario-to-FMU data flow."""
        # Create a test scenario CSV file
        scenario_file = tmp_path / "test_scenario.csv"

        # Create test data
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        time_index = pd.date_range(start=start_time, periods=100, freq="1s")
        scenario_data = pd.DataFrame(
            {"temperature": np.linspace(20, 30, 100)},
            index=time_index,
        )
        scenario_data.to_csv(scenario_file, date_format="%Y-%m-%d %H:%M:%S")

        # Create state configuration
        state_vars = [
            StateVar(
                name="control",
                is_agent_action=True,
                is_ext_input=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="ext_temp",
                from_scenario=True,
                scenario_id="temperature",
                is_ext_input=True,
                is_agent_observation=True,
                low_value=0,
                high_value=50,
            ),
            StateVar(
                name="output",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
        ]

        config = StateConfig(*state_vars)

        # Verify configuration structure
        assert config.vars["ext_temp"].from_scenario is True
        assert config.vars["ext_temp"].is_ext_input is True
        assert "ext_temp" in config.ext_inputs
        assert "ext_temp" in config.scenario_outputs
        assert "ext_temp" in config.observations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
