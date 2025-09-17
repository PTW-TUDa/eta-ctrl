import pytest

from examples.kea_tank.main import experiment


class TestKeaExample:
    @pytest.mark.disable_logging
    def test_main(self):
        overwrite = {"settings": {"log_to_file": False}}
        experiment(overwrite=overwrite)
