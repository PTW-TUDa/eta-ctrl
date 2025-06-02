import pytest

from eta_ctrl.agents.math_solver import MathSolver
from eta_ctrl.common import NoPolicy
from eta_ctrl.config import ConfigOptRun
from test.resources.agents.mpc_basic_env import MathSolverEnv


class TestMathSolver:
    @pytest.fixture(scope="class")
    def mpc_basic_env(self, temp_dir):
        config_run = ConfigOptRun(
            series="MPC_Basic_test_2023",
            name="test_mpc_basic",
            description="",
            path_root=temp_dir / "root",
            path_results=temp_dir / "results",
        )

        # Create the environment
        env = MathSolverEnv(
            env_id=1,
            config_run=config_run,
            prediction_horizon=10,
            scenario_time_begin="2021-12-01 06:00",
            scenario_time_end="2021-12-01 07:00",
            episode_duration=1800,
            sampling_time=1,
            model_parameters={},
        )
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def mpc_agent(self, mpc_basic_env):
        # set up the agent
        return MathSolver(NoPolicy, mpc_basic_env)

    def test_mpc_save_load(self, mpc_basic_env, mpc_agent, temp_dir):
        # save
        path = temp_dir / "test_mpc_basic_agent.zip"
        mpc_agent.save(path)

        # Load the agent from the saved file
        loaded_agent = MathSolver.load(path=path, env=mpc_basic_env)

        assert isinstance(loaded_agent, MathSolver)
        assert isinstance(loaded_agent.policy, NoPolicy)

        # Compare attributes before and after loading
        assert loaded_agent.model == mpc_agent.model
        assert loaded_agent.observation_space == mpc_agent.observation_space
        assert loaded_agent.num_timesteps == mpc_agent.num_timesteps

    def test_mpc_learn(self, mpc_agent):
        assert mpc_agent.learn(total_timesteps=5) is not None
        assert isinstance(mpc_agent, MathSolver)
