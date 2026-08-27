from __future__ import annotations

import abc
import json
import pathlib
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import asdict

from eta_ctrl.util import log_add_filehandler

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from eta_ctrl.config import Config, RunInfo

log = getLogger(__name__)


def log_to_file(config: Config, run_info: RunInfo) -> None:
    """Log output in terminal to the run_info file.

    :param config: Configuration to figure out the logging settings.
    :param run_info: Configuration for this optimization run.
    """
    file_path = run_info.log_output_path

    if config.settings.log_to_file:
        try:
            log_add_filehandler(filename=file_path)
        except Exception:
            log.exception("Log file could not be created.")


def log_run_config(config: Config, run_info: RunInfo) -> None:
    """Export the ``Config`` and ``RunInfo`` values to the run info JSON file.

    The output is a single JSON object whose keys are the union of
    the attributes from ``config`` and ``run_info`` (with ``config`` taking
    precedence on key conflicts).

    :param config: Current Config object.
    :param run_info: Current RunInfo object.
    """
    with run_info.run_info_path.open("w") as f:

        class Encoder(json.JSONEncoder):
            def default(self, o: object) -> object:
                if isinstance(o, pathlib.Path):
                    return str(o)
                if isinstance(o, abc.ABCMeta):
                    return None
                return repr(o)

        try:
            json.dump({**asdict(run_info), **config.model_dump()}, f, indent=4, cls=Encoder)
            log.info("Log file successfully created.")
        except TypeError:
            log.warning("Log file could not be created because of non-serializable input in config.")


def log_net_arch(model: BaseAlgorithm, run_info: RunInfo) -> None:
    """Store network architecture or policy information in a file. This requires for the model to be initialized,
    otherwise it will raise a ValueError.

    :param model: The algorithm whose network architecture is stored.
    :param run_info: current RunInfo (which contains paths for the log files).
    :raises: ValueError.
    """
    from .sb3_extensions.policies import NoPolicy  # noqa: PLC0415

    if not run_info.net_arch_path.exists() and model.policy is not None and model.policy.__class__ is not NoPolicy:
        with pathlib.Path(run_info.net_arch_path).open("w") as f:
            f.write(str(model.policy))

        log.info(f"Net arch / Policy information store successfully in: {run_info.net_arch_path}.")
    elif run_info.net_arch_path.exists():
        log.info(f"Net arch / Policy information already exists in {run_info.net_arch_path}")
