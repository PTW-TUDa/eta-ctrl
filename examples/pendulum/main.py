from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from eta_ctrl import get_logger
from eta_ctrl.core import EtaCtrl

if TYPE_CHECKING:
    from typing import Any


def main() -> None:
    get_logger()
    root_path = get_path()

    conventional(root_path)
    machine_learning(root_path)


def conventional(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> None:
    """Perform a conventionally controlled experiment with the pendulum environment.
    This uses the pendulum_conventional config file.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    experiment = EtaCtrl(root_path, "config_conventional", overwrite, relpath_config=".")
    experiment.play("conventional_series", "run1")


def machine_learning(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> None:
    """Perform machine learning experiment with the pendulum environment.
    This uses the pendulum_learning config file.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    # --main--

    experiment = EtaCtrl(root_path, "config_learning", overwrite, relpath_config=".")
    experiment.learn("learning_series", "run1", reset=True)
    experiment.play("learning_series", "run1")
    # --main--


def get_path() -> pathlib.Path:
    """Get the path of this file.

    :return: Path to this file.
    """
    return pathlib.Path(__file__).parent


if __name__ == "__main__":
    import sys

    sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

    main()
