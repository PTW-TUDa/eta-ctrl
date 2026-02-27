from functools import partial

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_ctrl.envs.base_env import BaseEnv
from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager


@pytest.fixture
def scenario_df():
    start_date = pd.Timestamp("2026/01/01")
    seconds = 16
    index = pd.date_range(start_date, start_date + pd.Timedelta(seconds=seconds), freq="s")
    data = {"scen1": [2 * i for i in range(seconds + 1)]}  # 0, 2, 4, ..., 2*i
    return pd.DataFrame(index=index, data=data)


class TestScenarioVecEnv:
    @pytest.fixture
    def scenario_manager_factory(self, scenario_df, monkeypatch):
        def factory(use_random_time_slice: bool):
            def dummy_load_data(scenario_manager):
                scenario_manager.scenarios = scenario_df
                scenario_manager.total_df_length = len(scenario_df)

            monkeypatch.setattr(CsvScenarioManager, "load_data", dummy_load_data)
            return CsvScenarioManager(
                scenario_configs=[],
                start_time=scenario_df.index[0],
                end_time=scenario_df.index[-1],
                total_time=2,
                resample_time=1,
                use_random_time_slice=use_random_time_slice,
            )

        return factory

    @pytest.fixture
    def scenario_vec_env_factory(self, unified_env_factory):
        def factory(scenario_manager):
            def create_env(env_id: int) -> BaseEnv:
                return unified_env_factory(
                    env_type="base",
                    env_id=env_id,
                    state_config_type="scenario",
                    episode_duration=1,
                    sampling_time=1,
                    scenario_manager=scenario_manager,
                )

            return DummyVecEnv([partial(create_env, i) for i in range(2)])  # Two envs

        return factory

    def test_scenario_random(self, scenario_vec_env_factory, scenario_manager_factory):
        scenario_manager = scenario_manager_factory(use_random_time_slice=True)
        scenario_vec_env = scenario_vec_env_factory(scenario_manager=scenario_manager)
        seed = 42

        scenario_vec_env.seed(seed=seed)
        scenario_vec_env.reset()

        env0 = scenario_vec_env.envs[0]
        env1 = scenario_vec_env.envs[1]

        # RNG selects random time beginnings, thus different offsets
        assert env0._scenario_offset == 1
        assert env1._scenario_offset == 7
        assert env0.state["scen1"].item() == 2 * 1
        assert env1.state["scen1"].item() == 2 * 7

        # Let environments reset, start new episode
        scenario_vec_env.step(np.array([[], []]))
        assert env0.n_steps == 0
        assert env1.n_steps == 0

        assert env0._scenario_offset == 11
        assert env1._scenario_offset == 9
        assert env0.state["scen1"].item() == 2 * 11
        assert env1.state["scen1"].item() == 2 * 9

        # Start new run, episode(s) should have the same values
        scenario_vec_env.seed(seed=seed)
        scenario_vec_env.reset()

        assert env0._scenario_offset == 1
        assert env1._scenario_offset == 7
        assert env0.state["scen1"].item() == 2 * 1
        assert env1.state["scen1"].item() == 2 * 7

    def test_scenario_no_random(self, scenario_vec_env_factory, scenario_manager_factory):
        scenario_manager = scenario_manager_factory(use_random_time_slice=False)
        scenario_vec_env = scenario_vec_env_factory(scenario_manager=scenario_manager)
        seed = 42

        scenario_vec_env.seed(seed=seed)
        scenario_vec_env.reset()

        env0 = scenario_vec_env.envs[0]
        env1 = scenario_vec_env.envs[1]

        # No random values
        assert env0._scenario_offset == 0
        assert env1._scenario_offset == 0
        assert env0.state["scen1"].item() == 2 * 0
        assert env1.state["scen1"].item() == 2 * 0

        # Let environments reset, start new episode
        scenario_vec_env.step(np.array([[], []]))
        assert env0.n_steps == 0
        assert env1.n_steps == 0

        assert env0._scenario_offset == 0
        assert env1._scenario_offset == 0
        assert env0.state["scen1"].item() == 2 * 0
        assert env1.state["scen1"].item() == 2 * 0
