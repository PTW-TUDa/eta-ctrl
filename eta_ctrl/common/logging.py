from __future__ import annotations

import abc
import json
import pathlib
from typing import TYPE_CHECKING

from attrs import asdict

from eta_ctrl.util import log_add_filehandler

from .sb3_extensions.policies import NoPolicy

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from eta_ctrl.config import Config, ConfigRun
from logging import getLogger

log = getLogger(__name__)


def log_to_file(config: Config, config_run: ConfigRun) -> None:
    """Log output in terminal to the run_info file.

    :param config: Configuration to figure out the logging settings.
    :param config_run: Configuration for this optimization run.
    """
    file_path = config_run.path_log_output

    if config.settings.log_to_file:
        try:
            log_add_filehandler(filename=file_path)
        except Exception:
            log.exception("Log file could not be created.")


def log_run_info(config: Config, config_run: ConfigRun) -> None:
    """Save run configuration to the run_info file.

    :param config: Configuration for the framework.
    :param config_run: Configuration for this optimization run.
    """
    with config_run.path_run_info.open("w") as f:

        class Encoder(json.JSONEncoder):
            def default(self, o: object) -> object:
                if isinstance(o, pathlib.Path):
                    return str(o)
                if isinstance(o, abc.ABCMeta):
                    return None
                return repr(o)

        try:
            json.dump({**asdict(config_run), **asdict(config)}, f, indent=4, cls=Encoder)
            log.info("Log file successfully created.")
        except TypeError:
            log.warning("Log file could not be created because of non-serializable input in config.")


def log_net_arch(model: BaseAlgorithm, config_run: ConfigRun) -> None:
    """Store network architecture or policy information in a file. This requires for the model to be initialized,
    otherwise it will raise a ValueError.

    :param model: The algorithm whose network architecture is stored.
    :param config_run: Optimization run configuration (which contains info about the file to store info in).
    :raises: ValueError.
    """
    if not config_run.path_net_arch.exists() and model.policy is not None and model.policy.__class__ is not NoPolicy:
        with pathlib.Path(config_run.path_net_arch).open("w") as f:
            f.write(str(model.policy))

        log.info(f"Net arch / Policy information store successfully in: {config_run.path_net_arch}.")
    elif config_run.path_net_arch.exists():
        log.info(f"Net arch / Policy information already exists in {config_run.path_net_arch}")
