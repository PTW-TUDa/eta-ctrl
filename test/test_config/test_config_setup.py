import pytest
from pydantic import ValidationError
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from eta_ctrl.agents.mpc_agent import MpcAgent
from eta_ctrl.common import NoPolicy
from eta_ctrl.config.config import ConfigSetup
from eta_ctrl.envs.base_env import BaseEnv

config = {
    "setup": {
        "agent_import": "eta_ctrl.agents.MpcAgent",
        "environment_import": "eta_ctrl.envs.BaseEnv",
        "policy_import": "stable_baselines3.common.policies.ActorCriticPolicy",
        "vectorizer_import": "stable_baselines3.common.vec_env.SubprocVecEnv",
        "monitor_wrapper": True,
        "norm_wrapper_obs": True,
        "norm_wrapper_reward": True,
        "tensorboard_log": True,
    },
}


class TestConfigSetup:
    def test_default_values(self):
        wanted_values = {"agent_import", "environment_import"}
        config_setup = ConfigSetup(**{k: config["setup"][k] for k in config["setup"] if k in wanted_values})

        assert config_setup.agent_class is MpcAgent
        assert config_setup.environment_class is BaseEnv
        assert config_setup.vectorizer_class is DummyVecEnv
        assert config_setup.policy_class is NoPolicy
        assert config_setup.monitor_wrapper is False
        assert config_setup.norm_wrapper_obs is False
        assert config_setup.norm_wrapper_reward is False
        assert config_setup.tensorboard_log is False

    def test_all_values(self):
        config_setup = ConfigSetup(**config["setup"])

        assert config_setup.agent_class is MpcAgent
        assert config_setup.environment_class is BaseEnv
        assert config_setup.vectorizer_class is SubprocVecEnv
        assert config_setup.policy_class is ActorCriticPolicy
        assert config_setup.monitor_wrapper is True
        assert config_setup.norm_wrapper_obs is True
        assert config_setup.norm_wrapper_reward is True
        assert config_setup.tensorboard_log is True

    def test_str(self):
        config_setup = ConfigSetup(**config["setup"])
        str_repr = "ConfigSetup(env=BaseEnv, agent=MpcAgent)"
        assert str(config_setup) == str_repr


class TestConfigSetupFail:
    def test_wrong_class(self):
        params = {**config["setup"], "agent_import": "eta_ctrl.envs.BaseEnv"}
        msg = (
            "'eta_ctrl.envs.BaseEnv' resolved to <class 'eta_ctrl.envs.base_env.BaseEnv'>, "
            "which is not a subclass of BaseAlgorithm"
        )
        with pytest.raises(TypeError, match=msg):
            ConfigSetup(**params)

    def test_class_not_found(self):
        params = {**config["setup"], "agent_import": "eta_ctrl.envs.FooBar"}
        msg = "module 'eta_ctrl.envs' has no attribute 'FooBar'"
        with pytest.raises(AttributeError, match=msg):
            ConfigSetup(**params)

    def test_wrong_type(self):
        params = {**config["setup"], "norm_wrapper_obs": "foobar"}
        msg = "Input should be a valid boolean"
        with pytest.raises(ValidationError, match=msg):
            ConfigSetup(**params)
