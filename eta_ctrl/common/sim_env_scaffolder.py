from __future__ import annotations

import pathlib
from logging import getLogger

from eta_ctrl.simulators.fmu import FMUSimulator
from eta_ctrl.util.io_utils import get_unique_output_path, toml_export
from eta_ctrl.util.utils import snake_to_camel_case

log = getLogger(__name__)


class SimEnvScaffolder:
    @staticmethod
    def from_fmu(fmu_path: pathlib.Path | str, output_dir: pathlib.Path | str | None = None) -> None:
        """Create a complete SimEnv environment from an FMU file.

        This method creates both a Python environment class file and a TOML state config file
        from an FMU, providing a complete setup for FMU-based simulations.

        :param fmu_path: Full path to the FMU file (including filename and .fmu extension).
        :param output_dir: Directory where files should be created. If None, uses the same directory as the FMU file.
        """
        fmu_path = pathlib.Path(fmu_path)

        if not fmu_path.exists():
            msg = f"FMU file not found at {fmu_path}"
            log.error(msg)
            raise FileNotFoundError(msg)

        # Extract FMU name and determine output directory
        fmu_name = fmu_path.stem
        output_directory = fmu_path.parent if output_dir is None else pathlib.Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Generate class name (convert to PascalCase)
        class_name = snake_to_camel_case(fmu_name) + "Env"

        # Create Python environment file
        python_file_path = output_directory / f"{fmu_name}.py"
        SimEnvScaffolder._create_environment_file(python_file_path, class_name, fmu_name)

        # Use custom output directory if output_dir is defined, default otherwise
        state_config_file_path = (
            output_directory / f"{fmu_name}_env_state_config.toml" if output_dir is not None else None
        )
        # Create state config TOML file
        SimEnvScaffolder.export_fmu_state_config(fmu_path=fmu_path, output_path=state_config_file_path)

    @staticmethod
    def export_fmu_state_config(fmu_path: pathlib.Path | str, output_path: pathlib.Path | str | None = None) -> None:
        """Export FMU input and output variables to a TOML file.

        This method extracts the input and output variables from the FMU model description
        and exports them to a TOML file for later use in other tasks.

        :param fmu_path: Full path to the FMU file (including filename and .fmu extension).
        :param output_path: Full path where the TOML file should be saved (including filename).
                            If None, saves to the same directory as the FMU file with name
                            '{fmu_name}_state_config.toml'.
        """
        with FMUSimulator.inspect(fmu_path) as ctx:
            fmu_data = {
                "fmu_info": {
                    "name": ctx["fmu_name"],
                    "path": str(ctx["fmu_path"]),
                },
                "actions": ctx["actions"],
                "observations": ctx["observations"],
            }

            base_path = (
                ctx["fmu_path"].parent / f"{ctx['fmu_name']}_env_state_config.toml"
                if output_path is None
                else pathlib.Path(output_path)
            )

        final_output_path = get_unique_output_path(base_path)
        toml_export(final_output_path, fmu_data)
        log.info(f"FMU variables exported to {final_output_path}")

    @staticmethod
    def export_fmu_parameters(fmu_path: pathlib.Path | str, output_path: pathlib.Path | str | None = None) -> None:
        """Export FMU parameter variables to a TOML file.

        This method extracts only the parameter variables from the FMU model description
        and exports them to a TOML file compatible with the environment_specific section
        of the global config file format.

        :param fmu_path: Full path to the FMU file (including filename and .fmu extension).
        :param output_path: Full path where the TOML file should be saved (including filename).
                            If None, saves to the same directory as the FMU file with name '{fmu_name}_parameters.toml'.
        """
        with FMUSimulator.inspect(fmu_path) as ctx:
            fmu_data = {
                "fmu_info": {
                    "name": ctx["fmu_name"],
                    "path": ctx["fmu_path"].name,
                },
                "parameters": ctx["parameters"],
            }

            base_path = (
                ctx["fmu_path"].parent / f"{ctx['fmu_name']}_parameters.toml"
                if output_path is None
                else pathlib.Path(output_path)
            )

        final_output_path = get_unique_output_path(base_path)
        toml_export(final_output_path, fmu_data)
        log.info(f"FMU parameters exported to {final_output_path}")

    @staticmethod
    def _create_environment_file(file_path: pathlib.Path, class_name: str, fmu_name: str) -> None:
        """Create a Python file with a SimEnv subclass.

        :param file_path: Path where the Python file should be created.
        :param class_name: Name of the environment class.
        :param fmu_name: Name of the FMU file (without extension).
        """
        # Check for overwrite protection
        if file_path.exists():
            log.warning(f"Python file already exists at {file_path}. Skipping creation.")
            return

        # Get template file path from envs module
        templates_dir = pathlib.Path(__file__).parent.parent / "envs" / "templates"
        template_path = templates_dir / "sim_env_template.py"

        # Read template content
        template_content = template_path.read_text(encoding="utf-8")

        # Replace template placeholders
        file_content = template_content.replace("SimEnvTemplate", class_name)
        file_content = file_content.replace("TEMPLATE_FMU_NAME", fmu_name)

        # Write the file
        file_path.write_text(file_content, encoding="utf-8")
        log.info(f"Created environment file: {file_path}")
