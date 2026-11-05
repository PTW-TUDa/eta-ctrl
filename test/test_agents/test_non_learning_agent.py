from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_ctrl.agents import NonLearningAgent


class DummyAgent(NonLearningAgent):
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.zeros((self.get_env().num_envs, *self.action_space.shape), dtype=np.float32), state


@pytest.fixture
def agent():
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    yield DummyAgent(policy=None, env=env, actions_order=["torque"])
    env.close()


class TestNonLearningAgent:
    def test_learn_logs_error_and_returns_agent(self, agent, caplog):
        assert agent.learn(total_timesteps=5) is agent
        assert "cannot learn" in caplog.text

    def test_save_not_supported(self, agent, tmp_path):
        with pytest.raises(NotImplementedError, match="cannot be saved"):
            agent.save(tmp_path / "model")
        assert not list(tmp_path.iterdir())

    def test_load_not_supported(self, tmp_path):
        with pytest.raises(NotImplementedError, match="cannot be loaded"):
            DummyAgent.load(tmp_path / "model")

    def test_no_policy(self, agent):
        assert not hasattr(agent, "policy")

    def test_get_parameters_not_supported(self, agent):
        with pytest.raises(NotImplementedError, match="has no parameters"):
            agent.get_parameters()
