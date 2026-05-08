from __future__ import annotations

import pytest

from scripts import pyomo_export_cli


@pytest.fixture(autouse=True)
def _disable_cli_logging_reconfiguration(monkeypatch):
    monkeypatch.setattr(pyomo_export_cli, "get_logger", lambda **_: None)


class TestDeriveModelName:
    def test_removes_model_suffix(self):
        assert pyomo_export_cli._derive_model_name("pkg.some_model.DrKeaModel") == "dr_kea"

    def test_keeps_name_when_no_suffix(self):
        assert pyomo_export_cli._derive_model_name("pkg.some_model.HeatPump") == "heat_pump"


class TestExportPyomoDataCli:
    def test_calls_create_state_with_expected_arguments(self, monkeypatch):
        calls = {}

        def fake_create_state(model_import, model_name, output_dir, **kwargs):
            calls["model_import"] = model_import
            calls["model_name"] = model_name
            calls["output_dir"] = output_dir
            calls["kwargs"] = kwargs

        monkeypatch.setattr(pyomo_export_cli.PyomoModel, "create_state", fake_create_state)
        monkeypatch.setattr(
            "sys.argv",
            [
                "export_pyomo_data",
                "examples.kea_tank.kea_pyomo_model.DrKeaModel",
                "--model-name",
                "dr_kea",
                "-o",
                "out_dir",
            ],
        )

        pyomo_export_cli.export_pyomo_data()

        assert calls["model_import"] == "examples.kea_tank.kea_pyomo_model.DrKeaModel"
        assert calls["model_name"] == "dr_kea"
        assert calls["output_dir"] == "out_dir"
        assert calls["kwargs"] == {}

    def test_uses_derived_defaults_when_optional_args_omitted(self, monkeypatch):
        calls = {}

        def fake_create_state(model_import, model_name, output_dir, **kwargs):
            calls["model_import"] = model_import
            calls["model_name"] = model_name
            calls["output_dir"] = output_dir
            calls["kwargs"] = kwargs

        monkeypatch.setattr(pyomo_export_cli.PyomoModel, "create_state", fake_create_state)
        monkeypatch.setattr(
            "sys.argv",
            [
                "export_pyomo_data",
                "examples.kea_tank.kea_pyomo_model.DrKeaModel",
            ],
        )

        pyomo_export_cli.export_pyomo_data()

        assert calls["model_import"] == "examples.kea_tank.kea_pyomo_model.DrKeaModel"
        assert calls["model_name"] == "dr_kea"
        assert calls["output_dir"] is None
        assert calls["kwargs"] == {}
