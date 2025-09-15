import gymnasium
import numpy as np
import pytest
from gymnasium.vector.utils import concatenate, create_empty_array
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_ctrl.common import NoPolicy
from eta_ctrl.util.julia_utils import julia_extensions_available
from test.resources.agents.rule_based import RuleBasedController

if julia_extensions_available():
    pass


class TestRuleBased:
    @pytest.fixture(scope="class")
    def vec_env(self):
        env = DummyVecEnv([lambda: gymnasium.make("CartPole-v1")])
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def rb_agent(self, vec_env):
        # Initialize the agent and create an instance of the policy and assign it to the policy attribute
        return RuleBasedController(policy=NoPolicy, env=vec_env)

    def test_rb_save_load(self, vec_env, rb_agent, temp_dir):
        # Save the agent
        path = temp_dir / "test_rule_based_agent.zip"
        rb_agent.save(path)

        # Load the agent from the saved file
        loaded_agent = RuleBasedController.load(path=path, env=vec_env)

        assert isinstance(loaded_agent, RuleBasedController)
        assert isinstance(loaded_agent.policy, NoPolicy)

        # Compare attributes before and after loading
        assert loaded_agent.observation_space == rb_agent.observation_space
        assert loaded_agent.num_timesteps == rb_agent.num_timesteps
        assert loaded_agent.state == rb_agent.state

    def test_rb_learn(self, rb_agent):
        assert rb_agent.learn(total_timesteps=5) is not None
        assert isinstance(rb_agent, RuleBasedController)

    def create_samples(self, space: gymnasium.spaces.Space, n: int = 1) -> np.ndarray | dict[str, np.ndarray]:
        samples = [space.sample() for _ in range(n)]
        empty_array = create_empty_array(space, n=n)
        return concatenate(space, samples, empty_array)

    def test_predict(self, rb_agent: RuleBasedController):
        obs = self.create_samples(rb_agent.env.observation_space, n=1)
        actions = rb_agent.predict(obs)
        assert actions[0] in (0, 1)
