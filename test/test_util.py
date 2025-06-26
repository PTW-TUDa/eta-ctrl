import json
import logging
import re
from pathlib import Path

import pytest
import toml

from eta_ctrl.util import (
    dict_search,
    json_import,
    log_add_filehandler,
)
from eta_ctrl.util.io_utils import load_config
from test.resources.config.config_python import config as python_dict


def test_log_file_handler():
    log_path = Path("test_log.log")
    log = log_add_filehandler(log_path, level=3)
    log.info("Info")
    log.error("Error")

    with log_path.open() as f:
        log_content = f.read()

    assert "Info" not in log_content
    assert "Error" in log_content

    logging.shutdown()
    log.handlers.clear()
    log_path.unlink()


def test_log_file_handler_no_path(caplog):
    log = log_add_filehandler(None, level=3)

    assert "No filename specified for filehandler. Using default filename eta_ctrl" in caplog.text
    assert "eta_ctrl" in log.handlers[-1].baseFilename

    logging.shutdown()
    Path(log.handlers[-1].baseFilename).unlink()
    log.handlers.clear()


def test_dict_search():
    assert dict_search({"key": "value"}, "value") == "key"


def test_dict_search_fail():
    with pytest.raises(ValueError, match=r".*not specified in specified dictionary"):
        dict_search({}, "value")


def test_remove_comments_json():
    with Path(Path(__file__).parent / "resources/remove_comments/removed_comments.json").open() as f:
        control = json.load(f)

    assert json_import(Path(__file__).parent / "resources/remove_comments/with_comments.json") == control


# IO utils

config_names = ["config1", "config2", "config3", "config1.json", "config2.toml", "config3.yaml"]


@pytest.mark.parametrize("config_name", config_names)
def test_load_config_file(config_name: str, config_resources_path: Path):
    file_path = config_resources_path / config_name
    config = load_config(file=file_path)

    assert config == python_dict


def test_load_config_file_fail(config_resources_path: Path):
    file_path = config_resources_path / "no_configfile"
    error_msg = re.escape(f"Config file not found: {file_path}")
    with pytest.raises(FileNotFoundError, match=error_msg):
        load_config(file=file_path)


def test_load_config_file_fail2(config_resources_path: Path, monkeypatch):
    file_path = config_resources_path / "config2"
    monkeypatch.setattr(toml, "load", lambda _: ["settings"])
    error_msg = re.escape(f"Config file {file_path} must define a dictionary of options.")
    with pytest.raises(TypeError, match=error_msg):
        load_config(file=file_path)
