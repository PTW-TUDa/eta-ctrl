import json
import logging
import re
from datetime import timedelta
from pathlib import Path

import pytest
import toml

from eta_ctrl.util import (
    dict_search,
    json_import,
    log_add_filehandler,
)
from eta_ctrl.util.io_utils import load_config
from eta_ctrl.util.utils import is_divisible, timestep_to_seconds, timestep_to_timedelta
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


class TestTimestep:
    REF_VALUE = 99.0
    REF_TIMEDELTA = timedelta(seconds=REF_VALUE)

    @pytest.mark.parametrize(
        ("timestep", "expected_seconds"),
        [
            (REF_VALUE, REF_VALUE),
            (timedelta(seconds=REF_VALUE), REF_VALUE),
            (int(REF_VALUE), REF_VALUE),
            (str(REF_VALUE), REF_VALUE),
            (timedelta(seconds=REF_VALUE, milliseconds=500), REF_VALUE + 0.5),
        ],
    )
    def test_timestep_to_seconds(self, timestep, expected_seconds):
        res = timestep_to_seconds(timestep=timestep)
        assert res == expected_seconds

    @pytest.mark.parametrize(
        ("timestep", "expected_timedelta"),
        [
            (REF_VALUE, REF_TIMEDELTA),
            (timedelta(seconds=REF_VALUE), REF_TIMEDELTA),
            (int(REF_VALUE), REF_TIMEDELTA),
            (str(REF_VALUE), REF_TIMEDELTA),
            (timedelta(seconds=REF_VALUE, milliseconds=500), REF_TIMEDELTA + timedelta(milliseconds=500)),
        ],
    )
    def test_timestep_to_timedelta(self, timestep, expected_timedelta):
        res = timestep_to_timedelta(timestep=timestep)
        assert res == expected_timedelta


class TestIsDivisible:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (15, 0.05),
            (15, 0.5),
            (15, 5),
            (12, 0.1),
            (1.2, 0.1),
            (90, 0.01),
            (90, 0.1),
            (90, 1),
        ],
    )
    def test_is_divisible(self, a, b):
        assert is_divisible(a=a, b=b)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (15, 0.0499),
            (0.12, 0.1),
            (90.000001, 0.1),
        ],
    )
    def test_is_not_divisible(self, a, b):
        assert not is_divisible(a=a, b=b)
