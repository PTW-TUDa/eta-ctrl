from unittest.mock import MagicMock

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector.utils import concatenate, create_empty_array
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_ctrl.common import NoPolicy
from test.resources.agents.rule_based import RuleBasedController


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

    # Reusable Environment Class
    class EmptyEnv(gymnasium.Env):
        """Reusable environment for testing empty observation spaces."""

        def __init__(self, obs_space_type="dict"):
            super().__init__()
            if obs_space_type == "dict":
                self.observation_space = spaces.Dict({})
                self.action_space = spaces.Dict({})
            else:  # box
                self.observation_space = spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float32)
                self.action_space = spaces.Box(low=np.array([]), high=np.array([]), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if isinstance(self.observation_space, spaces.Dict):
                return {}, {}
            return np.array([]), {}

        def step(self, action):
            obs, info = self.reset()
            return obs, 0.0, False, False, info

    def test_predict_empty_observations(self):
        """Test that RuleBased agent handles empty Dict observation spaces correctly.

        This tests the fix for the bug where the controller fails when no observations
        are defined, which occurs in AutomaticMode when control logic is inside the FMU.

        Key behavior: control_rules IS called with empty observations for each environment.
        """
        # Create empty Dict environment
        vec_env = DummyVecEnv([lambda: self.EmptyEnv(obs_space_type="dict")])

        try:
            # Use the actual RuleBasedController
            agent = RuleBasedController(policy=NoPolicy, env=vec_env)

            # Mock control_rules to return empty action and verify it's called
            agent.control_rules = MagicMock(return_value=np.array([]))

            # Test predict with empty Dict observations
            empty_obs = {}
            actions, state = agent.predict(empty_obs)

            # Assertions
            assert actions is not None, "Actions must not be None"
            assert state is None, "State should be None for RuleBased agent"
            # control_rules SHOULD be called with empty observation (explicit over implicit)
            agent.control_rules.assert_called_once_with({})

        finally:
            vec_env.close()

    def test_predict_empty_box_observations(self):
        """Test empty Box observation space with zero-length arrays.

        Note: Box spaces with shape (0,) ARE iterable (they produce one environment
        with an empty array). This is different from empty Dict spaces.
        control_rules WILL be called with the empty array.
        """
        # Create empty Box environment
        vec_env = DummyVecEnv([lambda: self.EmptyEnv(obs_space_type="box")])

        try:
            agent = RuleBasedController(policy=NoPolicy, env=vec_env)

            # Mock control_rules to return empty action
            agent.control_rules = MagicMock(return_value=np.array([]))

            # Test predict with empty Box observation
            empty_obs = np.array([[]])  # Shape (1, 0)
            actions, state = agent.predict(empty_obs)

            # Assertions
            assert actions is not None, "Actions must not be None"
            assert isinstance(actions, np.ndarray), "Actions must be numpy array"
            assert state is None, "State should be None for RuleBased agent"
            # For Box spaces, control_rules IS called (one env with empty array)
            agent.control_rules.assert_called_once()

        finally:
            vec_env.close()
