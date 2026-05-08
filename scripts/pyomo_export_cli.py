"""Command-line interface utilities for Pyomo model operations."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

from eta_ctrl import get_logger
from eta_ctrl.simulators import PyomoModel


def _ensure_project_root_on_syspath() -> None:
    """Ensure the project root is on sys.path so local model modules can be imported.

    Uses the location of this file (``scripts/pyomo_export_cli.py``) to find the
    project root reliably, regardless of the current working directory.
    Inserts unconditionally so Windows path-string variations cannot cause a
    false "already present" result from a naive string comparison.
    """
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.insert(0, project_root)


def _derive_model_name(model_import: str) -> str:
    """Create a default model name from the imported class name."""
    class_name = model_import.rsplit(".", 1)[-1]
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    return snake_case.removesuffix("_model")


def export_pyomo_data() -> None:
    """Command-line interface for exporting pyomo model data (state config and parameters) to TOML files."""
    # Initialize project logging
    get_logger(level=20, log_format="simple")  # INFO level for CLI output
    _ensure_project_root_on_syspath()

    parser = argparse.ArgumentParser(
        description="Export state config and parameters from a PyomoModel class", prog="export_pyomo_data"
    )
    parser.add_argument("model_import", type=str, help="Dotted import path to the PyomoModel subclass")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional output model name prefix (default: derived from model class name)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files (default: current directory)",
    )

    args = parser.parse_args()
    model_name = args.model_name or _derive_model_name(args.model_import)

    PyomoModel.create_state(
        args.model_import,
        model_name,
        args.output_dir,
    )


if __name__ == "__main__":
    export_pyomo_data()
