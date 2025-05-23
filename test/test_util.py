import json
import logging
import pathlib

import pytest

from eta_ctrl.util import (
    dict_search,
    get_logger,
    json_import,
    log_add_filehandler,
)


def test_log_file_handler():
    log_path = pathlib.Path("test_log.log")
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
    pathlib.Path(log.handlers[-1].baseFilename).unlink()
    log.handlers.clear()


def test_log_name_deprecation_warning():
    msg = "The 'name' argument is deprecated and will be removed in future versions."
    with pytest.warns(DeprecationWarning, match=msg):
        log = get_logger(name="test")
    # Remove actions of get_logger()
    log.propagate = True
    log.handlers.clear()


def test_dict_search():
    assert dict_search({"key": "value"}, "value") == "key"


def test_dict_search_fail():
    with pytest.raises(ValueError, match=r".*not specified in specified dictionary"):
        dict_search({}, "value")


def test_remove_comments_json():
    with pathlib.Path(pathlib.Path(__file__).parent / "resources/remove_comments/removed_comments.json").open() as f:
        control = json.load(f)

    assert json_import(pathlib.Path(__file__).parent / "resources/remove_comments/with_comments.json") == control
