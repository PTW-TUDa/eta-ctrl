import asyncio
import logging
import pathlib
import random
import shutil

import pytest
from _pytest.monkeypatch import MonkeyPatch

from test.test_envs.base_test_classes import (
    # Unified factory fixtures
    config_run_factory as config_run_factory,
    state_config_factory as state_config_factory,
    temp_directory_factory as temp_directory_factory,
    unified_env_factory as unified_env_factory,
)


@pytest.fixture(autouse=True, scope="session")
def _silence_logging():
    logging.root.setLevel(logging.ERROR)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Check for the disable_logging marker
    root_level = logging.CRITICAL if "disable_logging" in item.keywords else logging.NOTSET
    # Set logging level to INFO if caplog is used
    eta_ctrl_level = logging.INFO if "caplog" in item.fixturenames else logging.ERROR

    # Set disable logging level for root logger
    logging.disable(root_level)
    # Set logger level for "eta_ctrl" namespace
    logging.getLogger("eta_ctrl").setLevel(eta_ctrl_level)


@pytest.fixture(scope="session")
def temp_dir():
    while True:
        temp_dir = pathlib.Path.cwd() / f"tmp_{random.randint(10000, 99999)}"
        try:
            temp_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        else:
            break

    yield temp_dir
    shutil.rmtree(temp_dir)


async def stop_execution(sleep_time):
    await asyncio.sleep(sleep_time)


@pytest.fixture(scope="class")
def class_monkeypatch():
    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session")
def root_path():
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(scope="session")
def resources_path(root_path):
    return root_path / "test" / "resources"


@pytest.fixture(scope="session")
def config_live_connect(resources_path):
    """Test configuration for live connect."""
    return {"file": resources_path / "config_live_connect.json"}


@pytest.fixture(scope="session")
def config_fmu(resources_path):
    """Test configuration for FMU simulator."""
    return {"file": resources_path / "damped_oscillator/damped_oscillator.fmu"}


@pytest.fixture(scope="session")
def agent_resources_path(resources_path):
    return resources_path / "agents"


@pytest.fixture(scope="session")
def config_resources_path(resources_path):
    return resources_path / "config"
