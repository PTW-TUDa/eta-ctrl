import logging

from examples.damped_oscillator.main import (
    experiment_conventional as ex_oscillator,
    get_path as get_oscillator_path,
)
from examples.pendulum.main import (
    conventional as ex_pendulum_conventional,
    get_path as get_pendulum_path,
    machine_learning as ex_pendulum_learning,
)


class TestPendulumExample:
    def test_conventional(self, tmp_path):
        ex_pendulum_conventional(
            get_pendulum_path(),
            {
                "paths": {"results_relpath": tmp_path / "results"},
                "settings": {
                    "log_to_file": False,
                    "environment": {"do_render": False},
                },
            },
        )

    def test_learning(self, tmp_path):
        ex_pendulum_learning(
            get_pendulum_path(),
            {
                "paths": {"results_relpath": tmp_path / "results"},
                "setup": {
                    # SubprocVecEnv spawns Windows subprocesses which deadlock in pytest and
                    # keep model/log files open after learn(), causing PermissionError in play().
                    # DummyVecEnv runs everything in-process, which is correct for testing.
                    "vectorizer_import": "stable_baselines3.common.vec_env.DummyVecEnv",
                    "tensorboard_log": False,
                },
                "settings": {
                    "episode_duration": 0.2,
                    "n_episodes_learn": 1,
                    "n_episodes_play": 1,
                    "save_model_every_x_episodes": 10,
                    "n_environments": 1,
                    "log_to_file": False,
                    "agent": {
                        "n_steps": 4,
                        "batch_size": 2,
                        "n_epochs": 1,
                        "policy_kwargs": {"net_arch": [8]},
                    },
                    "environment": {"do_render": False},
                },
            },
        )


class TestOscillatorExample:
    def test_oscillator(self, tmp_path):
        try:
            ex_oscillator(
                get_oscillator_path(),
                {
                    "paths": {"results_relpath": tmp_path / "results"},
                    "settings": {"log_to_file": False},
                },
            )
        finally:
            logging.shutdown()
