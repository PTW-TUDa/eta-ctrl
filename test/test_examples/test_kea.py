import pytest

from examples.kea_tank.main import experiment as base_experiment
from examples.kea_tank.main_pyomo_sim import experiment as pyomo_sim_experiment


class TestKeaExample:
    @pytest.mark.disable_logging
    def test_base_main(self, tmp_path):
        overwrite = {
            "paths": {"results_relpath": tmp_path / "results"},
            "settings": {"log_to_file": False, "episode_duration": 60, "prediction_horizon": 60},
        }
        base_experiment(overwrite=overwrite)

    @pytest.mark.disable_logging
    def test_pyomo_sim_main(self, tmp_path):
        overwrite = {
            "paths": {"results_relpath": tmp_path / "results"},
            "settings": {"log_to_file": False, "episode_duration": 60, "prediction_horizon": 60},
        }
        pyomo_sim_experiment(overwrite=overwrite)
