from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from eta_ctrl import EtaCtrl, get_logger

if TYPE_CHECKING:
    from typing import Any


def main() -> None:
    get_logger(level=1, log_format="logname")

    experiment()


def experiment(overwrite: dict[str, Any] | None = None) -> None:
    """Perform a conventionally controlled experiment with the cleaning machine environment.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    root_path = pathlib.Path(__file__).parent
    experiment = EtaCtrl(root_path=root_path, config_overwrite=overwrite, config_relpath=".", config_name="config")
    experiment.play(series_name="cleaning_machine", run_name="test_run")


if __name__ == "__main__":
    main()
