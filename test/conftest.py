import asyncio
import logging
import pathlib
import random
import shutil

import pytest


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


@pytest.fixture(scope="session")
def config_live_connect():
    """Test configuration for live connect."""
    return {"file": pathlib.Path(__file__).parent / "resources/config_live_connect.json"}


@pytest.fixture(scope="session")
def config_fmu():
    """Test configuration for FMU simulator."""
    return {"file": pathlib.Path(__file__).parent / "resources/damped_oscillator/damped_oscillator.fmu"}


@pytest.fixture(scope="session")
def config_resources_path():
    return pathlib.Path(__file__).parent / "resources" / "agents"
