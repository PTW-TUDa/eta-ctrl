import pathlib

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_ctrl.common import NoPolicy
from eta_ctrl.config import ConfigOptRun
from eta_ctrl.envs import NoVecEnv
from eta_ctrl.util.julia_utils import julia_extensions_available

if julia_extensions_available():
    from eta_ctrl.agents.nsga2 import Nsga2
    from eta_ctrl.envs.julia_env import JuliaEnv


@pytest.mark.skipif(not julia_extensions_available(), reason="PyJulia installation required!")
class TestNSGA2:
    scenario_time_begin = "2017-01-24 00:01"
    scenario_time_end = "2017-01-24 23:59"
    episode_duration = 1800
    sampling_time = 1

    def config_run(self, temp_dir):
        directory = pathlib.Path(temp_dir) if not isinstance(temp_dir, pathlib.Path) else temp_dir

        return ConfigOptRun(
            series="Nsga2_test_2023",
            name="test_nsga2",
            description="",
            path_root=directory / "root",
            path_results=directory / "results",
        )

    def create_stored_agent_file(self, path, temp_dir):
        """Execute this function directly when necessary to refresh the stored NSGA2 agent model file. This function
        creates a new NSGA2 model and saves the result in the given path.

        ..code-block: console

            python -c "import tempfile; from test.test_agents import TestNSGA2; cls = TestNSGA2();
            cls.create_stored_agent_file('test/resources/agents/', tempfile.TemporaryDirectory().name)"

        :param path: Path where the julia environment file is located. This path is also used to store the trained
                     model file.
        :param temp_dir: Path to store result files during training of the model.
        """
        directory = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

        env = JuliaEnv(
            env_id=1,
            config_run=self.config_run(temp_dir),
            scenario_time_begin=self.scenario_time_begin,
            scenario_time_end=self.scenario_time_end,
            episode_duration=self.episode_duration,
            sampling_time=self.sampling_time,
            julia_env_file=directory.absolute() / "Nsga2Env.jl",
            is_multithreaded=True,
        )
        agent = Nsga2(
            policy=NoPolicy,
            env=NoVecEnv([lambda: env]),
            population=1000,
            crossovers=0.3,
            n_generations=2,
            max_retries=1000000,
            predict_learn_steps=10,
            seed=2139846,
        )
        agent.learn(10)
        agent.save(directory.absolute() / "test_nsga2_agent.zip")

    @pytest.fixture(scope="class")
    def julia_env(self, config_resources_path, temp_dir):
        env = JuliaEnv(
            env_id=1,
            config_run=self.config_run(temp_dir),
            scenario_time_begin=self.scenario_time_begin,
            scenario_time_end=self.scenario_time_end,
            episode_duration=self.episode_duration,
            sampling_time=self.sampling_time,
            julia_env_file=config_resources_path / "Nsga2Env.jl",
            is_multithreaded=True,
        )
        no_vec_env = NoVecEnv([lambda: env])
        yield no_vec_env
        no_vec_env.close()

    @pytest.fixture(scope="class")
    def loaded_agent(self, config_resources_path, julia_env):
        path = config_resources_path / "test_nsga2_agent.zip"
        return Nsga2.load(path=path, env=julia_env)

    @pytest.fixture(scope="class")
    def default_agent(self, julia_env):
        return Nsga2(env=julia_env, policy=NoPolicy)

    def test_load_agent_setup(self, loaded_agent):
        assert isinstance(loaded_agent, Nsga2)
        assert isinstance(loaded_agent.env, DummyVecEnv)

    @pytest.mark.parametrize(
        ("attr", "value"),
        [
            ("population", 1000),
            ("mutations", 0.05),
            ("crossovers", 0.3),
            ("n_generations", 2),
            ("max_cross_len", 1),
            ("max_retries", 1000000),
            ("predict_learn_steps", 10),
            ("seed", 2139846),
            ("tensorboard_log", None),
            ("event_params", 60),
        ],
    )
    def test_load_agent_attributes(self, attr, value, loaded_agent):
        assert getattr(loaded_agent, attr) == value

    def test_predict(self, loaded_agent):
        observations = loaded_agent.env.reset()
        action, state = loaded_agent.predict(observations)

        events, variables = action

        assert len(action["events"] == 60)
        assert len(action["variables"] == 60)
        assert state is None
        assert isinstance(events, np.ndarray)
        assert events.shape == (60,)
        assert isinstance(variables, np.ndarray)
        assert variables.shape == (60,)

    @pytest.mark.parametrize(
        ("attr", "value"),
        [
            ("population", 100),
            ("mutations", 0.05),
            ("crossovers", 0.1),
            ("n_generations", 100),
            ("max_cross_len", 1),
            ("max_retries", 100000),
            ("predict_learn_steps", 5),
            ("seed", 42),
        ],
    )
    def test_default_agent_attributes(self, attr, value, default_agent):
        assert getattr(default_agent, attr) == value

    def test_setup_learn(self, default_agent):
        init_total_timesteps = 10
        total_timesteps, callback = default_agent._setup_learn(
            total_timesteps=init_total_timesteps,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="predict",
            progress_bar=False,
        )
        assert total_timesteps == init_total_timesteps
        assert callback is not None
        assert default_agent.num_timesteps == 0
        assert default_agent.ep_info_buffer is not None
        assert default_agent.start_time is not None
