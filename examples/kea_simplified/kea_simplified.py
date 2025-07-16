from eta_ctrl.envs.sim_env import SimEnv

SimEnv.export_fmu_state_config("examples/kea_simplified/kea_simplified.fmu")

SimEnv.export_fmu_parameters("examples/kea_simplified/kea_simplified.fmu")
