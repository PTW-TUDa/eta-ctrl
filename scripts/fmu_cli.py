"""Command-line interface utilities for FMU operations."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from eta_ctrl.common.sim_env_scaffolder import SimEnvScaffolder

# Configure logging at module level for all CLI scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for CLI output
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _validate_fmu_path(fmu_path: pathlib.Path) -> None:
    """Validate that the FMU path exists and has correct extension."""
    if not fmu_path.exists():
        print(f"Error: FMU file not found at {fmu_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    if fmu_path.suffix.lower() != ".fmu":
        print(f"Error: File must have .fmu extension, got {fmu_path.suffix}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


def create_sim_env() -> None:
    """Command-line interface for creating a SimEnv environment from an FMU."""
    parser = argparse.ArgumentParser(description="Create a SimEnv environment from an FMU file", prog="create_sim_env")
    parser.add_argument("fmu_path", type=str, help="Path to the FMU file (including filename and .fmu extension)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files (default: same as FMU directory)",
    )

    args = parser.parse_args()
    fmu_path = pathlib.Path(args.fmu_path)
    _validate_fmu_path(fmu_path)

    SimEnvScaffolder.from_fmu(fmu_path, args.output_dir)


def export_fmu_data() -> None:
    """Command-line interface for exporting FMU data (state config and parameters) to TOML files."""
    parser = argparse.ArgumentParser(
        description="Export FMU state config and parameters to TOML files", prog="export_fmu_data"
    )
    parser.add_argument("fmu_path", type=str, help="Path to the FMU file (including filename and .fmu extension)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for TOML files (default: same as FMU directory)",
    )

    args = parser.parse_args()
    fmu_path = pathlib.Path(args.fmu_path)
    _validate_fmu_path(fmu_path)

    # Export state config
    state_config_output = None
    if args.output_dir:
        state_config_output = str(pathlib.Path(args.output_dir) / f"{fmu_path.stem}_env_state_config.toml")

    SimEnvScaffolder.export_fmu_state_config(fmu_path, state_config_output)

    # Export parameters
    parameters_output = None
    if args.output_dir:
        parameters_output = str(pathlib.Path(args.output_dir) / f"{fmu_path.stem}_parameters.toml")

    SimEnvScaffolder.export_fmu_parameters(fmu_path, parameters_output)
