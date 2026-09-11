from eta_ctrl.common.sim_env_scaffolder import SimEnvScaffolder

SimEnvScaffolder.export_fmu_state_config("examples/fmu_export_demo/kea_simplified.fmu")
SimEnvScaffolder.export_fmu_parameters("examples/fmu_export_demo/kea_simplified.fmu")
