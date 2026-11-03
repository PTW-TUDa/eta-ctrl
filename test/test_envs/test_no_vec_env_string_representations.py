"""Tests for __str__ and __repr__ of NoVecEnv."""

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

from eta_ctrl.envs.no_vec_env import NoVecEnv


class _MultithreadedEnv(gymnasium.Env):
    """Minimal environment which satisfies the NoVecEnv requirements."""

    is_multithreaded = True
    action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def __repr__(self) -> str:
        return "_MultithreadedEnv()"


class TestNoVecEnvStringRepresentations:
    """Tests for NoVecEnv.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def vec_env(self):
        env = NoVecEnv([_MultithreadedEnv, _MultithreadedEnv, _MultithreadedEnv])
        yield env
        env.close()

    def test_str(self, vec_env):
        assert str(vec_env) == "NoVecEnv(_MultithreadedEnv, 1 instance, 3 environment(s))"

    def test_repr(self, vec_env):
        assert repr(vec_env) == "NoVecEnv(env=_MultithreadedEnv(), num_envs=3)"
