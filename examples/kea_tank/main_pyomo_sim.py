from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

from eta_ctrl import EtaCtrl, get_logger

if TYPE_CHECKING:
    from typing import Any


def main() -> None:
    get_logger(level=logging.INFO, log_format="logname")

    experiment()


def experiment(overwrite: dict[str, Any] | None = None) -> None:
    """Perform a conventionally controlled experiment with the cleaning machine environment.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    root_path = pathlib.Path(__file__).parent
    env_path = root_path / "environments"
    overwrite = {}

    # Adapt config to PyomoSimEnv
    overwrite["setup"] = {"environment_import": "examples.kea_tank.environments.kea_pyomo_sim_env.DrKeaPyomoSimEnv"}
    overwrite["paths"] = {"state_file_relpath": "pyomo_sim_state_config.toml"}
    environment_specific = {
        "path_env": env_path.absolute(),
        "model_parameters": {
            "p_heat": 10,  # kW, heating power consumption
            "tank_temperature_start": 60,  # °C
            "tank_temperature_min": 55,  # °C
            "tank_temperature_max": 65,  # °C
            "temperature_change_heating": 0.02,  # Kelvin per second
            "temperature_change_cleaning": -0.01,  # Kelvin per second
        },
    }
    overwrite["settings"] = {"environment": environment_specific}
    experiment = EtaCtrl(root_path=root_path, config_overwrite=overwrite, config_relpath=".", config_name="config.toml")

    experiment.play(series_name="kea_tank_pyomo_sim", run_name="example_run")


if __name__ == "__main__":
    main()
