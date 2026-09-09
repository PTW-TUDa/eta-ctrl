from __future__ import annotations

from unittest.mock import MagicMock, patch

from eta_ctrl.envs.live_env import LIVE_READ_RETRY_ATTEMPTS, LIVE_READ_RETRY_DELAY_SECONDS


def test_empty_result_retries_read_after_delay(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.return_value = {"out1": 1.0, "out2": 2.0}

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        results = environment._retry_read_if_invalid({})

    expected_ids = [environment.state_config.map_ext_ids[name] for name in environment.state_config.ext_outputs]
    assert results == {"out1": 1.0, "out2": 2.0}
    environment.connection_manager.read.assert_called_once_with(*expected_ids)
    sleep_mock.assert_called_once_with(LIVE_READ_RETRY_DELAY_SECONDS)


def test_valid_result_does_not_retry(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    results = {"out1": 1.0}

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        assert environment._retry_read_if_invalid(results) is results

    environment.connection_manager.read.assert_not_called()
    sleep_mock.assert_not_called()


def test_empty_retry_result_is_returned_after_ten_attempts(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.return_value = {}

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        assert environment._retry_read_if_invalid({}) == {}

    assert environment.connection_manager.read.call_count == LIVE_READ_RETRY_ATTEMPTS
    assert sleep_mock.call_count == LIVE_READ_RETRY_ATTEMPTS
    sleep_mock.assert_called_with(LIVE_READ_RETRY_DELAY_SECONDS)


def test_retry_stops_after_first_non_empty_read(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.side_effect = [{}, {}, {"out1": 1.0}]

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        results = environment._retry_read_if_invalid({})

    assert results == {"out1": 1.0}
    assert environment.connection_manager.read.call_count == 3
    assert sleep_mock.call_count == 3


def test_nan_result_retries_read_after_delay(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.return_value = {"out1": 1.0, "out2": 2.0}

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        results = environment._retry_read_if_invalid({"out1": float("nan"), "out2": 2.0})

    assert results == {"out1": 1.0, "out2": 2.0}
    environment.connection_manager.read.assert_called_once()
    sleep_mock.assert_called_once_with(LIVE_READ_RETRY_DELAY_SECONDS)


def test_retry_continues_after_connection_error(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.side_effect = [
        ConnectionError("Host timeout"),
        {"out1": 1.0, "out2": 2.0},
    ]

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        results = environment._retry_read_if_invalid({"out1": float("nan")})

    assert results == {"out1": 1.0, "out2": 2.0}
    assert environment.connection_manager.read.call_count == 2
    assert sleep_mock.call_count == 2


def test_step_retries_empty_connection_manager_result(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.step.return_value = {}
    environment.connection_manager.read.return_value = {"out1": 1.0, "out2": 2.0}
    environment.get_external_inputs = MagicMock(return_value={"input": 3.0})
    environment.set_external_outputs = MagicMock()

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        result = environment._step()

    assert result == (0, False, False, {})
    environment.connection_manager.step.assert_called_once_with(value={"input": 3.0})
    environment.connection_manager.read.assert_called_once()
    sleep_mock.assert_called_once_with(LIVE_READ_RETRY_DELAY_SECONDS)
    environment.set_external_outputs.assert_called_once_with(external_outputs={"out1": 1.0, "out2": 2.0})


def test_step_retries_when_connection_manager_raises(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.step.side_effect = ConnectionError("Host timeout")
    environment.connection_manager.read.return_value = {"out1": 1.0, "out2": 2.0}
    environment.get_external_inputs = MagicMock(return_value={"input": 3.0})
    environment.set_external_outputs = MagicMock()

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        result = environment._step()

    assert result == (0, False, False, {})
    environment.connection_manager.step.assert_called_once_with(value={"input": 3.0})
    environment.connection_manager.read.assert_called_once()
    sleep_mock.assert_called_once_with(LIVE_READ_RETRY_DELAY_SECONDS)
    environment.set_external_outputs.assert_called_once_with(external_outputs={"out1": 1.0, "out2": 2.0})


def test_reset_retries_empty_initial_read(unified_env_factory) -> None:
    environment = unified_env_factory(env_type="live", state_config_type="live")
    environment.connection_manager = MagicMock()
    environment.connection_manager.read.side_effect = [{}, {"out1": 1.0, "out2": 2.0}]
    environment._init_connection_manager = MagicMock()
    environment.set_external_outputs = MagicMock()

    with patch("eta_ctrl.envs.live_env.sleep") as sleep_mock:
        result = environment._reset()

    assert result == {}
    assert environment.connection_manager.read.call_count == 2
    sleep_mock.assert_called_once_with(LIVE_READ_RETRY_DELAY_SECONDS)
    environment.set_external_outputs.assert_called_once_with(external_outputs={"out1": 1.0, "out2": 2.0})
