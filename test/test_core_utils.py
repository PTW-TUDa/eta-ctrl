"""Tests for the reset/backup behaviour of :func:`~eta_ctrl.core_utils.vectorize_environment` and

``EtaCtrl._prepare_model``.

Also covers the regression test for the ``EtaCtrl._prepare_model(reset=True)`` bug where an existing model file
could never be backed up (``model_path / "..."`` raised ``NotADirectoryError`` because ``model_path`` is a file).
See test/test_util.py for tests of the underlying :func:`~eta_ctrl.util.io_utils.backup_file` helper.
"""

import pytest
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize

from eta_ctrl.core_utils import vectorize_environment
from test.test_envs.base_test_classes import TestBaseEnv


def obs_mean(envs: VecNormalize) -> float:
    """Return one scalar observation running mean, regardless of whether the obs space is a Box or a Dict."""
    obs_rms = envs.obs_rms
    if isinstance(obs_rms, dict):
        obs_rms = next(iter(obs_rms.values()))
    assert isinstance(obs_rms, RunningMeanStd)
    return obs_rms.mean[0]


def set_obs_mean(envs: VecNormalize, value: float) -> None:
    """Set every observation running mean to ``value``, for a Box or a Dict obs space."""
    obs_rms = envs.obs_rms
    rms_list = obs_rms.values() if isinstance(obs_rms, dict) else [obs_rms]
    for rms in rms_list:
        rms.mean[:] = value


class TestVectorizeEnvironmentReset:
    """Tests that vectorize_environment() respects the reset flag for normalization data."""

    @pytest.fixture
    def run_info(self, run_info_factory):
        info, _ = run_info_factory()
        info.create_results_folders()
        return info

    @pytest.fixture
    def env_settings(self, state_config_factory):
        return {
            "state_config": state_config_factory("default"),
            "episode_duration": 7200,
            "sampling_time": 300,
        }

    def _vectorize(self, run_info, env_settings, *, reset=False) -> VecNormalize:
        envs = vectorize_environment(
            env=TestBaseEnv,
            run_info=run_info,
            env_settings=dict(env_settings),
            callback=lambda _env: None,
            training=True,
            reset=reset,
            norm_wrapper_obs=True,
        )
        assert isinstance(envs, VecNormalize)
        return envs

    def test_reset_false_loads_existing_normalization_data(self, run_info, env_settings):
        """No reset, normalization data exists. Should reload."""
        envs = self._vectorize(run_info, env_settings)
        set_obs_mean(envs, 42.0)
        envs.save(str(run_info.vec_normalize_path))
        envs.close()

        reloaded = self._vectorize(run_info, env_settings, reset=False)
        try:
            assert obs_mean(reloaded) == pytest.approx(42.0)
            assert run_info.vec_normalize_path.is_file()
        finally:
            reloaded.close()

    def test_reset_true_backs_up_and_creates_fresh_normalization(self, run_info, env_settings):
        """Reset, normalization data exists. Should not reload."""
        envs = self._vectorize(run_info, env_settings)
        set_obs_mean(envs, 42.0)
        envs.save(str(run_info.vec_normalize_path))
        envs.close()

        fresh = self._vectorize(run_info, env_settings, reset=True)
        try:
            assert obs_mean(fresh) == pytest.approx(0.0)
            assert not run_info.vec_normalize_path.is_file()

            backups = list(
                run_info.vec_normalize_path.parent.glob(
                    f"{run_info.vec_normalize_path.stem}_*.bak{run_info.vec_normalize_path.suffix}"
                )
            )
            assert len(backups) == 1
        finally:
            fresh.close()

    def test_reset_true_without_existing_data_creates_fresh_wrapper(self, run_info, env_settings):
        """Reset, normalization data does not exist."""
        assert not run_info.vec_normalize_path.is_file()

        fresh = self._vectorize(run_info, env_settings, reset=True)
        try:
            assert obs_mean(fresh) == pytest.approx(0.0)
        finally:
            fresh.close()
