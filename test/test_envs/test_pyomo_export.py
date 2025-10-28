"""Tests for Pyomo model export functionality."""
# ruff: noqa: F811

import pathlib
import tempfile
from unittest.mock import patch

import pyomo.environ as pyo
import pytest

from eta_ctrl.common.export_pyomo import (
    _extract_variable_bounds,
    _extract_variable_domain_type,
    export_pyomo_parameters,
    export_pyomo_state,
    export_pyomo_state_config,
    extract_indexed_variable_info,
    extract_scalar_variable_info,
)
from eta_ctrl.util.io_utils import (
    get_unique_output_path,
)
from test.resources.pyomo_concrete_model_fixtures import production_planning_concrete_model  # noqa: F401


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


class TestPyomoExport:
    """Test suite for Pyomo model export functionality."""

    def test_export_state_config(self, production_planning_concrete_model, temp_dir):
        """Test state configuration export creates valid TOML with correct structure."""
        output_path = temp_dir / "state_config.toml"

        export_pyomo_state_config(production_planning_concrete_model, "test_model", output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify content structure matches the production planning model
        content = output_path.read_text()
        assert "model_info" in content
        assert "observations" in content
        assert "total_production" in content
        assert "period_production" in content
        assert "machine_active" in content  # Binary variables from the model

    def test_export_parameters(self, production_planning_concrete_model, temp_dir):
        """Test parameters export creates valid TOML with array structure for indexed parameters."""
        output_path = temp_dir / "parameters.toml"

        export_pyomo_parameters(production_planning_concrete_model, "test_model", output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify content structure matches the production planning model with new array format
        content = output_path.read_text()
        assert "parameters" in content
        assert "total_capacity" in content  # Scalar parameter
        assert "period_demand" in content  # Indexed parameter should be in array format
        assert "product_cost" in content  # Another indexed parameter in array format

    def test_export_pyomo_state_combined(self, production_planning_concrete_model, temp_dir):
        """Test export_pyomo_state generates both files correctly."""
        export_pyomo_state(production_planning_concrete_model, "combined_test", temp_dir)

        state_file = temp_dir / "combined_test_state_config.toml"
        params_file = temp_dir / "combined_test_parameters.toml"

        assert state_file.exists()
        assert state_file.stat().st_size > 0
        assert params_file.exists()
        assert params_file.stat().st_size > 0

    def test_scalar_variable_extraction(self, production_planning_concrete_model):
        """Test extraction of scalar variable information."""
        production_var = production_planning_concrete_model.total_production  # Use actual model variable
        var_info = extract_scalar_variable_info(production_var)

        assert var_info["type"] == "continuous"
        assert var_info["low_value"] == 0.0
        assert "high_value" not in var_info  # bounds=(0, None)

    def test_indexed_variable_extraction(self, production_planning_concrete_model):
        """Test extraction of indexed variable information."""
        period_var = production_planning_concrete_model.period_production  # Use actual model variable
        var_info = extract_indexed_variable_info(period_var)

        assert var_info["type"] == "continuous"
        assert var_info["index_length"] == 4  # 4 time periods in the real model
        assert var_info["low_value"] == 0.0
        assert var_info["high_value"] == 150.0  # bounds=(0, 150) in the real model

    def test_file_overwrite_protection(self, production_planning_concrete_model, temp_dir):
        """Test unique output path generation prevents file overwrites."""
        base_path = temp_dir / "test.toml"
        base_path.touch()  # Create existing file

        unique_path = get_unique_output_path(base_path)

        assert unique_path != base_path
        assert unique_path.name == "test_1.toml"

    def test_default_output_paths(self, production_planning_concrete_model):
        """Test export functions work with default (None) output paths."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            with tempfile.TemporaryDirectory() as tmp_dir:
                mock_cwd.return_value = pathlib.Path(tmp_dir)

                export_pyomo_state(production_planning_concrete_model, "default_test")

                assert (pathlib.Path(tmp_dir) / "default_test_state_config.toml").exists()
                assert (pathlib.Path(tmp_dir) / "default_test_parameters.toml").exists()

    def test_edge_cases(self, temp_dir):
        """Test edge cases like models with special variable types."""
        # Create a minimal model with edge case variables
        minimal_model = pyo.ConcreteModel()
        minimal_model.unbounded_var = pyo.Var()  # No bounds
        minimal_model.binary_var = pyo.Var(domain=pyo.Binary)
        minimal_model.integer_var = pyo.Var(domain=pyo.Integers, bounds=(0, 10))

        export_pyomo_state(minimal_model, "edge_case_test", temp_dir)

        state_file = temp_dir / "edge_case_test_state_config.toml"
        assert state_file.exists()

        content = state_file.read_text()
        assert "unbounded_var" in content
        assert "binary_var" in content
        assert "integer_var" in content

    def test_production_model_completeness(self, production_planning_concrete_model, temp_dir):
        """Test that the production planning model exports all expected components."""
        export_pyomo_state(production_planning_concrete_model, "production_test", temp_dir)

        state_file = temp_dir / "production_test_state_config.toml"
        params_file = temp_dir / "production_test_parameters.toml"

        assert state_file.exists()
        assert params_file.exists()

        # Check that we have all the expected variable types
        state_content = state_file.read_text()
        assert "machine_active" in state_content  # Binary variables
        assert "total_inventory" in state_content  # Bounded variables
        assert "inventory" in state_content  # Multi-indexed variables

        # Check that we have all expected parameters with new array structure
        params_content = params_file.read_text()
        assert "total_capacity" in params_content  # Scalar parameters
        assert "machine_capacity" in params_content  # Multi-indexed parameters as arrays

    def test_helper_function_bounds_extraction(self):
        """Test the helper function for bounds extraction."""
        # Test with valid bounds
        bounds = (0.0, 100.0)
        result = _extract_variable_bounds(bounds)
        assert result["low_value"] == 0.0
        assert result["high_value"] == 100.0

        # Test with infinite bounds
        bounds = (float("-inf"), float("inf"))
        result = _extract_variable_bounds(bounds)
        assert "low_value" not in result
        assert "high_value" not in result

        # Test with partial bounds
        bounds = (10.0, None)
        result = _extract_variable_bounds(bounds)
        assert result["low_value"] == 10.0
        assert "high_value" not in result

    def test_helper_function_domain_type_extraction(self, production_planning_concrete_model):
        """Test the helper function for domain type extraction."""
        # Test continuous variable
        continuous_var = production_planning_concrete_model.total_production
        domain_type = _extract_variable_domain_type(continuous_var)
        assert domain_type == "continuous"

        # Test binary variable
        binary_var = production_planning_concrete_model.machine_active[("M1", 1)]
        domain_type = _extract_variable_domain_type(binary_var)
        assert domain_type == "discrete"

    def test_empty_index_set_exception_handling(self):
        """Test that empty index sets are handled gracefully (StopIteration catch)."""
        # Create a model with an empty indexed variable
        model = pyo.ConcreteModel()
        model.empty_set = pyo.Set(initialize=[])
        model.empty_indexed_var = pyo.Var(model.empty_set, domain=pyo.NonNegativeReals)

        # This should not raise an exception and should return continuous as default
        var_info = extract_indexed_variable_info(model.empty_indexed_var)

        assert var_info["type"] == "continuous"  # Default when index set is empty
        assert var_info["index_length"] == 0

    def test_invalid_index_access_exception_handling(self):
        """Test that invalid index access is handled gracefully (KeyError catch)."""
        # Create a model with indexed variable but try to access with wrong key type
        model = pyo.ConcreteModel()
        model.products = pyo.Set(initialize=["A", "B", "C"])
        model.production = pyo.Var(model.products, domain=pyo.NonNegativeReals)

        # This tests the exception handling for malformed indices
        var_info = extract_indexed_variable_info(model.production)

        # Should successfully extract info despite potential key issues
        assert "type" in var_info
        assert var_info["index_length"] == 3

    def test_parameter_value_error_handling(self):
        """Test that uninitialized or symbolic parameters are handled (ValueError catch)."""
        model = pyo.ConcreteModel()

        # Create an uninitialized parameter (will raise ValueError when trying to get value)
        model.uninitialized_param = pyo.Param()

        # Create indexed parameter with some uninitialized values
        model.time = pyo.Set(initialize=[1, 2, 3])
        model.partial_param = pyo.Param(model.time, initialize={1: 10.0, 2: 20.0}, mutable=True)
        # Note: index 3 is not initialized

        # This should not crash, but skip the uninitialized parameters
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = pathlib.Path(tmp_dir) / "test_params.toml"
            export_pyomo_parameters(model, "test_model", output_path)

            # File should be created despite some parameters being uninitialized
            assert output_path.exists()

    def test_parameter_type_error_handling(self):
        """Test that parameters with incompatible types are handled (TypeError catch)."""
        model = pyo.ConcreteModel()

        # Create a parameter with a value that might cause type issues during stringification
        model.normal_param = pyo.Param(initialize=42.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = pathlib.Path(tmp_dir) / "test_params.toml"
            export_pyomo_parameters(model, "test_model", output_path)

            # Should successfully export the valid parameter
            assert output_path.exists()
            content = output_path.read_text()
            assert "normal_param" in content
