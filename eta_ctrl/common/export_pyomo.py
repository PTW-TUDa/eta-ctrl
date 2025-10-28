"""Pyomo model export utilities.

This module provides functions for exporting Pyomo model components to TOML files.
"""

from __future__ import annotations

import pathlib
from logging import getLogger
from typing import TYPE_CHECKING

from pyomo import environ as pyo

from eta_ctrl.util import toml_export
from eta_ctrl.util.io_utils import get_unique_output_path

if TYPE_CHECKING:
    from typing import Any

log = getLogger(__name__)


def _extract_variable_bounds(bounds: tuple) -> dict[str, Any]:
    """Extract and process variable bounds into a standardized format.

    This helper function handles the common logic for extracting low and high bounds
    from Pyomo variables, converting infinite bounds to None and handling edge cases.

    :param bounds: Tuple of (lower_bound, upper_bound) from Pyomo variable
    :return: Dictionary containing processed bound information
    """
    bounds_info = {}

    if bounds[0] is not None:
        low_val = float(bounds[0]) if bounds[0] != float("-inf") else None
        if low_val is not None:
            bounds_info["low_value"] = low_val

    if bounds[1] is not None:
        high_val = float(bounds[1]) if bounds[1] != float("inf") else None
        if high_val is not None:
            bounds_info["high_value"] = high_val

    return bounds_info


def _extract_variable_domain_type(variable: pyo.Var) -> str:
    """Extract the domain type from a Pyomo variable.

    This helper function determines whether a variable is continuous or discrete
    based on its domain attribute, with a fallback to continuous as the default.

    :param variable: Pyomo variable with domain attribute
    :return: String indicating "continuous" or "discrete"
    """
    if hasattr(variable, "domain") and variable.domain is not None:
        domain_name = str(variable.domain)
        return "continuous" if "Real" in domain_name else "discrete"
    return "continuous"  # Default assumption


def extract_indexed_variable_info(component: pyo.Var) -> dict[str, Any]:
    """Extract comprehensive information from indexed Pyomo variables.

    This function analyzes indexed Pyomo variables to extract domain information,
    bounds, and index set details. It handles edge cases where variables may not
    have complete information by providing reasonable defaults.

    :param component: Indexed Pyomo variable component to analyze.
    :return: Dictionary containing variable type, bounds, and index information.
    """
    var_info: dict[str, Any] = {}

    # Get domain and bounds from first index if available, reusing scalar extraction logic
    try:
        first_key = next(iter(component.index_set()))
        first_var = component[first_key]
        # Reuse extract_scalar_variable_info for consistent domain and bounds extraction
        var_info.update(extract_scalar_variable_info(first_var))
    except (StopIteration, KeyError):
        # StopIteration: Empty index set (no variables in the indexed component)
        # KeyError: Index access failed (malformed or inaccessible index)
        # In both cases, defaulting to continuous is legitimate since most Pyomo variables
        # are continuous unless explicitly specified as discrete/binary
        var_info["type"] = "continuous"  # Default assumption

    # Add index information for indexed variables - critical for reconstruction
    index_set = component.index_set()
    var_info["index_length"] = len(index_set) if hasattr(index_set, "__len__") else "unknown"

    # Set index_set name with proper handling of None values
    # This helps identify the relationship between variables and their indices
    if hasattr(index_set, "name") and index_set.name is not None:
        var_info["index_set"] = str(index_set.name)
    else:
        var_info["index_set"] = "unknown"

    return var_info


def extract_scalar_variable_info(component: pyo.Var) -> dict[str, Any]:
    """Extract comprehensive information from scalar Pyomo variables.

    This function analyzes scalar Pyomo variables to extract domain information
    and bounds. It provides fallback defaults for variables without explicit
    domain or bounds specifications.

    :param component: Scalar Pyomo variable component to analyze.
    :return: Dictionary containing variable type and bounds information.
    """
    var_info: dict[str, Any] = {}

    # Extract domain type using helper function
    var_info["type"] = _extract_variable_domain_type(component)

    # Add bounds information if available using helper function
    if hasattr(component, "bounds") and component.bounds != (None, None):
        var_info.update(_extract_variable_bounds(component.bounds))

    return var_info


def export_pyomo_state_config(model: pyo.ConcreteModel, model_name: str, output_path: pathlib.Path) -> None:
    """Export Pyomo model variables (observations) to a TOML file.

    This method extracts the variables from the Pyomo model and exports them to a TOML file
    for later use in state configuration.

    ATTENTION: All variables are treated as observations, you need to separate these.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model for identification.
    :param output_path: Full path where the TOML file should be saved (including filename).
    """
    # Extract variables (observations) from the model
    observations = []

    for component in model.component_objects(pyo.Var):
        var_name = component.name
        var_info = {
            "name": var_name,
            "is_indexed": component.is_indexed(),
        }

        # Extract variable-specific information
        if component.is_indexed():
            var_info.update(extract_indexed_variable_info(component))
        else:
            var_info.update(extract_scalar_variable_info(component))

        observations.append(var_info)

    pyomo_data = {
        "model_info": {
            "name": model_name,
            "type": "pyomo",
        },
        "observations": observations,
    }

    final_output_path = get_unique_output_path(output_path)
    toml_export(final_output_path, pyomo_data)
    log.info(f"Pyomo model variables exported to {final_output_path}")


def export_pyomo_parameters(model: pyo.ConcreteModel, model_name: str, output_path: pathlib.Path) -> None:
    """Export Pyomo model parameters to a TOML file.

    This method extracts parameter names and values from the Pyomo model and exports them to a TOML file.
    For indexed parameters, all values are collected as arrays to preserve the complete parameter information.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model for identification.
    :param output_path: Full path where the TOML file should be saved (including filename).
    """
    # Extract parameters from the model - preserve all values for indexed parameters
    parameters = {}

    for component in model.component_objects(pyo.Param):
        param_name = component.name

        if component.is_indexed():
            # For indexed parameters, collect all values as arrays to preserve complete information
            param_values = []
            param_indices = []

            for index in component.index_set():
                try:
                    value = pyo.value(component[index])
                    if value is not None:
                        param_values.append(str(value))
                        param_indices.append(str(index))
                except (ValueError, TypeError):
                    # ValueError: Parameter value cannot be evaluated (e.g., symbolic expressions,
                    #            uninitialized parameters, or mutable parameters without values)
                    # TypeError: Parameter index or value type incompatible with conversion
                    #           (e.g., complex objects that can't be stringified)
                    # Skip invalid entries but continue processing other indices
                    continue

            # Store as arrays if we have values
            if param_values:
                parameters[param_name] = {"values": param_values, "indices": param_indices, "is_indexed": True}
        else:
            # For scalar parameters, store the actual value
            try:
                value = pyo.value(component)
                if value is not None:
                    parameters[param_name] = {"value": str(value), "is_indexed": False}
            except (ValueError, TypeError):
                # ValueError: Parameter value cannot be evaluated (e.g., uninitialized parameter)
                # TypeError: Parameter value type incompatible with string conversion
                # Skip invalid parameters but continue processing others
                continue

    final_output_path = get_unique_output_path(output_path)

    pyomo_data = {
        "parameters": parameters,
        "model_info": {"name": model_name, "path": str(final_output_path), "type": "pyomo_parameters"},
    }

    toml_export(final_output_path, pyomo_data)
    log.info(f"Pyomo model parameters exported to {final_output_path}")
    log.info(f"Exported {len(parameters)} parameters with complete value arrays")


def export_pyomo_state(model: pyo.ConcreteModel, model_name: str, output_dir: pathlib.Path | str | None = None) -> None:
    """Export Pyomo model state config and parameters files.

    This is the main public interface for exporting Pyomo model data, creating both
    state configuration and parameters files.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model for identification.
    :param output_dir: Directory where files should be created. If None, uses current working directory.
    """
    # Centralize output directory logic
    output_directory = pathlib.Path.cwd().absolute() if output_dir is None else pathlib.Path(output_dir).absolute()
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create specific file paths
    state_config_path = output_directory / f"{model_name}_state_config.toml"
    parameters_path = output_directory / f"{model_name}_parameters.toml"

    # Call export functions with concrete paths
    export_pyomo_state_config(model, model_name, state_config_path)
    export_pyomo_parameters(model, model_name, parameters_path)

    log.info(f"Created Pyomo model files for '{model_name}' in {output_directory}")
