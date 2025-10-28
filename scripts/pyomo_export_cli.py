"""Command-line interface utilities for Pyomo model operations."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

from eta_ctrl import get_logger
from eta_ctrl.envs.pyomo_env import PyomoEnv


def _validate_env_path(env_path: pathlib.Path) -> None:
    """Validate that the PyomoEnv file exists and has correct extension."""
    if not env_path.exists():
        print(f"Error: PyomoEnv file not found at {env_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    if env_path.suffix.lower() != ".py":
        print(f"Error: File must have .py extension, got {env_path.suffix}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


def _load_pyomo_env_class(env_path: pathlib.Path) -> type[PyomoEnv]:
    """Load a PyomoEnv class from a Python file."""
    spec = importlib.util.spec_from_file_location("pyomo_env_module", env_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {env_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for PyomoEnv subclass
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, PyomoEnv) and attr is not PyomoEnv:
            return attr

    print(f"Error: No PyomoEnv subclass found in {env_path}", file=sys.stderr)  # noqa: T201
    sys.exit(1)


def export_pyomo_data() -> None:
    """Command-line interface for exporting pyomo model data (state config and parameters) to TOML files."""
    # Initialize project logging
    get_logger(level=20, log_format="simple")  # INFO level for CLI output

    parser = argparse.ArgumentParser(
        description="Create state config and parameters from a PyomoEnv instance", prog="export_pyomo_data"
    )
    parser.add_argument("env_path", type=str, help="Path to the Python file containing the PyomoEnv subclass")
    parser.add_argument("model_name", type=str, help="Name for the model")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files (default: current directory)",
    )

    args = parser.parse_args()
    env_path = pathlib.Path(args.env_path)
    _validate_env_path(env_path)

    # Load PyomoEnv class and create instance
    env_class = _load_pyomo_env_class(env_path)
    env_instance = env_class()  # type: ignore[call-arg]

    # Get ConcreteModel from the model property (calls _model() internally)
    model = env_instance.model[0]

    # Create state config from the model
    PyomoEnv.create_state(model, args.model_name, args.output_dir)
