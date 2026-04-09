from __future__ import annotations

import pathlib

from eta_ctrl.config.config import Config


def get_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def load_minimal() -> Config:
    return Config.from_file(root_path=get_path(), config_relpath=".", config_name="config_minimal")


def load_minimal_with_scenarios() -> Config:
    return Config.from_file(root_path=get_path(), config_relpath=".", config_name="config_minimal_with_scenarios")


def load_full() -> Config:
    return Config.from_file(root_path=get_path(), config_relpath=".", config_name="config_full")
