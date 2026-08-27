"""Pyomo model export utilities.

This module provides functions for generating ETA Ctrl config files from Pyomo model components.
Pyomo Variables and Parameters and exported as follows:

+------------------------+-------------+-------------+
|                        | indexed     | non-indexed |
+========================+=============+=============+
| Variables              | action      | None        |
+------------------------+-------------+-------------+
| mutable Parameters     | observation | observation |
+------------------------+-------------+-------------+
| immutable Parameters   | None        | config      |
+------------------------+-------------+-------------+

Actions and observations are saved in the state config.
Non-indexed immutable parameters are exported to the model_parameters section in the agent config.
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

    from pyomo.core.base.var import IndexedVar

log = getLogger(__name__)


def extract_indexed_variable_info(component: IndexedVar) -> dict[str, Any]:
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

    return var_info


def extract_scalar_variable_info(component: pyo.Var) -> dict[str, Any]:
    """Extract comprehensive information from scalar Pyomo variables.

    This function analyzes scalar Pyomo variables to extract domain information
    and bounds.

    :param component: Scalar Pyomo variable component to analyze.
    :return: Dictionary containing variable type and bounds information.
    """
    var_info: dict[str, Any] = {}

    lower_bound = component.lower
    if isinstance(lower_bound, (int, float)):
        low_val = float(lower_bound) if lower_bound != float("-inf") else None
        if low_val is not None:
            var_info["low_value"] = low_val

    upper_bound = component.upper
    if isinstance(upper_bound, (int, float)):
        high_val = float(upper_bound) if upper_bound != float("inf") else None
        if high_val is not None:
            var_info["high_value"] = high_val
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


# ---------------------------------------------------------------------------
# PyomoModel-specific export (actions / observations / model_parameters split)
# ---------------------------------------------------------------------------


def export_pyomo_model_state_config(model: pyo.ConcreteModel, model_name: str, output_path: pathlib.Path) -> None:
    """Export a PyomoModel's components to a state config TOML file.

    Classification rules:

    * Indexed ``pyo.Var`` components  → ``[[actions]]``
    * Mutable ``pyo.Param`` components → ``[[observations]]``

    This assumption that Variables are actions is not correct, but it is impossible
    to distinguish which Variables are actions and which are not.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model for identification.
    :param output_path: Full path (including filename) for the TOML file.
    """
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for component in model.component_objects(pyo.Var):
        if not component.is_indexed():
            continue
        # Reuse existing helper; strip index metadata that belongs to internal bookkeeping
        raw = extract_indexed_variable_info(component)
        var_info: dict[str, Any] = {"name": component.name, **raw}

        actions.append(var_info)

    for component in model.component_objects(pyo.Param):
        if not component.mutable:
            continue
        var_info = {"name": component.name}
        # Assume indexed and mutable parameter are scenario data
        if component.is_indexed():
            var_info.update({"duration": "n_prediction_steps", "from_scenario": True})
            log.info("Found %s as indexed observation parameter. Set 'from_scenario' to False if applicable", component)
        observations.append(var_info)

    pyomo_data: dict[str, Any] = {}
    if actions:
        pyomo_data["actions"] = actions
    if observations:
        pyomo_data["observations"] = observations

    final_output_path = get_unique_output_path(output_path)
    toml_export(final_output_path, pyomo_data)
    log.info(f"PyomoModel state config exported to {final_output_path}")


def export_pyomo_model_parameters(model: pyo.ConcreteModel, model_name: str, output_path: pathlib.Path) -> None:
    """Export immutable scalar Pyomo parameters to a model parameters TOML file.

    The output corresponds to the ``[settings.agent.model_parameters]`` section
    of a run config.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model for identification.
    :param output_path: Full path (including filename) for the TOML file.
    """
    model_parameters: dict[str, Any] = {}

    for component in model.component_objects(pyo.Param):
        if component.is_indexed() or component.mutable:
            continue
        value = component.default()
        # If no value is provided, it will default to 'NoValue'
        if value is pyo.Param.NoValue:
            value = "PLACEHOLDER"
        model_parameters[component.name] = value

    pyomo_data: dict[str, Any] = {
        "model_info": {"name": model_name, "type": "pyomo_model_parameters"},
        "model_parameters": model_parameters,
    }

    final_output_path = get_unique_output_path(output_path)
    toml_export(final_output_path, pyomo_data)
    log.info(
        f"PyomoModel parameters exported to {final_output_path}. "
        "Place these values under [settings.agent.model_parameters] in your experiment config."
    )


def export_pyomo_model_state(
    model: pyo.ConcreteModel, model_name: str, output_dir: pathlib.Path | str | None = None
) -> None:
    """Export a PyomoModel's state config and model parameters to TOML files.

    This is the main public interface for :class:`~eta_ctrl.simulators.PyomoModel`
    state generation. It writes two files:

        * ``{model_name}_state_config.toml`` — indexed Vars as actions and mutable
            Params as observations.
        * ``{model_name}_model_parameters.toml`` — immutable scalar Params that belong in
            ``[settings.agent.model_parameters]`` of the run config.

    :param model: Pyomo ConcreteModel instance.
    :param model_name: Name of the model used for file naming.
    :param output_dir: Target directory. Defaults to the current working directory.
    """
    output_directory = pathlib.Path.cwd().absolute() if output_dir is None else pathlib.Path(output_dir).absolute()
    output_directory.mkdir(parents=True, exist_ok=True)
    export_pyomo_model_state_config(model, model_name, output_directory / f"{model_name}_state_config.toml")
    export_pyomo_model_parameters(model, model_name, output_directory / f"{model_name}_model_parameters.toml")

    log.info(f"Created PyomoModel files for '{model_name}' in {output_directory}")
