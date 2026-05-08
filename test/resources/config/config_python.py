config = {  # noqa: INP001
    "setup": {
        "environment_import": "eta_ctrl.envs.PyomoSimEnv",
        "agent_import": "eta_ctrl.agents.MpcAgent",
        "vectorizer_import": "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv",
        "policy_import": "eta_ctrl.common.NoPolicy",
    },
    "paths": {"state_file_relpath": "config/test_env_state_config"},
    "settings": {
        "sampling_time": 10,
        "episode_duration": 1800,
        "n_episodes_play": 1,
        "n_environments": 1,
        "verbose": 2,
        "seed": 123,
        "scenario_time_begin": "2022-03-18 00:00",
        "scenario_time_end": "2022-03-18 00:50",
        "prediction_horizon": 1200,
        "scenario_files": [
            {
                "path": "electricity_price_test.csv",
                "interpolation_method": "ffill",
            }
        ],
        "environment": {
            "sim_steps_per_sample": 1,
            "model_parameters": {
                "N": 5,
                "n_start": 1,
                "S": 1200,
                "p_heat": 10,
                "p_int": 1,
                "p_clean": 4,
                "p_dry": 8,
                "durationStart": 10,
                "durationCleaning": 600,
                "durationDrying": 120,
                "durationLoading": 240,
                "tankTemperatureMin": 55,
                "tankTemperatureMax": 65,
                "tankTemperatureStart": 60,
                "temperatureChangeHeatingValue": 0.033,
                "temperatureChangeCleaningValue": -0.025,
            },
        },
        "agent": {"action_index": 1, "solver_name": "cplex_direct"},
    },
}
