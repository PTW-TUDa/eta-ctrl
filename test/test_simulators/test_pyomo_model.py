from __future__ import annotations

import pathlib
import re
import tempfile

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from eta_ctrl.simulators.pyomo_model import PyomoModel


class _SimpleModel(PyomoModel):
    """Minimal PyomoModel for testing: one scalar and one indexed mutable parameter."""

    def _model(self) -> pyo.AbstractModel:
        m = pyo.AbstractModel()
        m.t = pyo.RangeSet(0, self.n_prediction_steps)
        m.scalar_param = pyo.Param(within=pyo.Reals, mutable=True, initialize=0.0)
        m.indexed_param = pyo.Param(m.t, within=pyo.Reals, mutable=True, initialize=0.0)
        return m


class _InitFailsModel(PyomoModel):
    """Model used to assert export path does not call subclass __init__."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        msg = "__init__ should not be called during export"
        raise RuntimeError(msg)

    def _model(self) -> pyo.AbstractModel:
        m = pyo.AbstractModel()
        m.t = pyo.RangeSet(0, self.n_prediction_steps)
        m.scalar_param = pyo.Param(within=pyo.Reals)
        m.state = pyo.Var(m.t, within=pyo.Reals, bounds=(0, 1))
        return m


@pytest.fixture(scope="module")
def model():
    return _SimpleModel(sampling_time=10, prediction_horizon=60)


class TestPyomoModel:
    def test_missing_prediction_horizon(self):
        msg = "Prediction_horizon parameter is not present in config."
        with pytest.raises(ValueError, match=msg):
            PyomoModel(sampling_time=1)

    def test_not_divisible_prediction_horizon(self):
        msg = re.escape(
            "The sampling_time must fit evenly into the prediction_horizon "
            "(prediction_horizon % sampling_time must equal 0)."
        )
        with pytest.raises(ValueError, match=msg):
            PyomoModel(sampling_time=2, prediction_horizon=3)

    def test_missing_model(self):
        with pytest.raises(NotImplementedError):
            PyomoModel(sampling_time=1, prediction_horizon=4)

    def test_missing_start_value_mapping(self, model):
        msg = "Tried to access 'self._start_value_mapping' from '_SimpleModel', but it doesn't exist."
        with pytest.raises(AttributeError, match=msg):
            model.start_value_mapping  # noqa: B018


class TestPyoInitParams:
    def test_empty_returns_empty_dict(self, model: _SimpleModel) -> None:
        assert model._pyo_init_params() == {}

    def test_wraps_scalar_value_in_pyomo_format(self, model: _SimpleModel) -> None:
        model.model_parameters = {"scalar_param": 5.0}
        assert model._pyo_init_params() == {None: {"scalar_param": {None: 5.0}}}

    def test_converts_inf_string_to_float(self, model: _SimpleModel) -> None:
        model.model_parameters = {"scalar_param": "inf"}
        result = model._pyo_init_params()
        assert result == {None: {"scalar_param": {None: float("inf")}}}


class TestPyoUpdateParams:
    scalar_value = 42.0
    indexed_values = list(range(7))

    ####################
    ### Scalar tests ###
    ####################
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (scalar_value, scalar_value),
            ([scalar_value], scalar_value),
            (np.array([42]), scalar_value),
            (True, 1),
        ],
    )
    def test_update_scalar(self, model: _SimpleModel, value, expected) -> None:
        model.pyo_update_params({"scalar_param": value})
        assert pyo.value(model.model.scalar_param) == expected

    def test_update_scalar_fail(self, model: _SimpleModel) -> None:
        msg = re.escape("Received non-scalar value [42, 43] for component 'scalar_param'")
        with pytest.raises(TypeError, match=msg):
            model.pyo_update_params({"scalar_param": [42, 43]})

    #####################
    ### Indexed tests ###
    #####################
    @pytest.mark.parametrize(
        ("values"),
        [
            indexed_values,
            np.array(indexed_values),
            pd.Series(data=indexed_values),
        ],
    )
    def test_updates_indexed_sequence(self, model: _SimpleModel, values) -> None:
        model.pyo_update_params({"indexed_param": values})
        for i in range(len(values)):
            assert pyo.value(model.model.indexed_param[i]) == values[i]

    @pytest.mark.parametrize(
        ("values"),
        [
            {i: i for i in range(7)},
        ],
    )
    def test_update_indexed_mapping(self, model: _SimpleModel, values) -> None:
        model.pyo_update_params({"indexed_param": values})
        for i in range(len(values)):
            assert pyo.value(model.model.indexed_param[i]) == values[i]

    def test_update_indexed_fail(self, model: _SimpleModel) -> None:
        msg = re.escape("Received unsupported datatype <class 'set'> for component 'indexed_param'")
        with pytest.raises(TypeError, match=msg):
            model.pyo_update_params({"indexed_param": {42, 43}})

    def test_update_indexed_not_enough_fail(self, model: _SimpleModel) -> None:
        msg = re.escape("Component 'indexed_param' needs 7 values but 2 were supplied.")
        with pytest.raises(ValueError, match=msg):
            model.pyo_update_params({"indexed_param": [42, 43]})

    def test_update_indexed_value_scalar(self, model: _SimpleModel) -> None:
        model.pyo_update_params({"indexed_param": [self.scalar_value]})
        assert pyo.value(model.model.indexed_param[0]) == self.scalar_value

    def test_update_and_get_solution(self, model: _SimpleModel) -> None:
        model.pyo_update_params({"indexed_param": self.indexed_values, "scalar_param": self.scalar_value})
        indexed_solution, scalar_solution = model.pyo_get_solution()
        assert indexed_solution["indexed_param"] == self.indexed_values
        assert scalar_solution["scalar_param"] == self.scalar_value


class TestPyomoModelEnvCompatibility:
    @pytest.fixture
    def model(self):
        class _SimplePyomoSimModel(PyomoModel):
            _start_value_mapping = {"scalar_param": "expression"}

            def _model(self) -> pyo.AbstractModel:
                m = pyo.AbstractModel()
                m.t = pyo.RangeSet(0, self.n_prediction_steps)
                m.scalar_param = pyo.Param(within=pyo.Reals, mutable=True, initialize=0.0)
                m.expression = pyo.Expression(m.t)
                return m

        return _SimplePyomoSimModel(sampling_time=1, prediction_horizon=1)

    def test_missing_ext_output(self, model: PyomoModel):
        msg = "Missing 'foo' in start_value_mapping of '_SimplePyomoSimModel'"
        with pytest.raises(KeyError, match=msg):
            model.check_pyomo_sim_compatibility(ext_outputs=["foo"])

    @pytest.mark.parametrize("com", ["scalar_param", "expression"])
    def test_missing_expr_component(self, model: PyomoModel, com):
        delattr(model.model, com)
        msg = f"Component {com} does not exist in '_SimplePyomoSimModel'"
        with pytest.raises(ValueError, match=msg):
            model.check_pyomo_sim_compatibility(ext_outputs=["scalar_param"])

    @pytest.mark.parametrize(("com", "com_type"), [("scalar_param", "Param"), ("expression", "Expression")])
    def test_wrong_component(self, model: PyomoModel, com, com_type):
        setattr(model.model, com, pyo.Var(model.model.t))
        msg = f"Component {com} must be of type '{com_type}', but is 'IndexedVar'"
        with pytest.raises(TypeError, match=msg):
            model.check_pyomo_sim_compatibility(ext_outputs=["scalar_param"])

    def test_wrong_component_index(self, model: PyomoModel):
        model.model.scalar_param = pyo.Param(model.model.t)
        msg = "Component scalar_param must not be indexed, use 'ScalarParam' instead."
        with pytest.raises(TypeError, match=msg):
            model.check_pyomo_sim_compatibility(ext_outputs=["scalar_param"])

    def test_wrong_expr_component_index(self, model: PyomoModel):
        model.model.expression = pyo.Expression()
        msg = "Component expression must be indexed to retrieve the second value."
        with pytest.raises(TypeError, match=msg):
            model.check_pyomo_sim_compatibility(ext_outputs=["scalar_param"])


# ---------------------------------------------------------------------------
# Shared constants for kea_tank integration tests
# ---------------------------------------------------------------------------

_KEA_IMPORT = "examples.kea_tank.kea_pyomo_model.DrKeaModel"
_KEA_PARAMS: dict = {
    "p_heat": 10.0,
    "tank_temperature_start": 60.0,
    "tank_temperature_min": 55.0,
    "tank_temperature_max": 65.0,
    "temperature_change_heating": 0.02,
    "temperature_change_cleaning": -0.01,
}
_KEA_KWARGS: dict = {
    "sampling_time": 10.0,
    "prediction_horizon": 60.0,
    "model_parameters": _KEA_PARAMS,
}


class TestLoadFromImport:
    """Tests for PyomoModel.load_from_import."""

    def test_returns_pyomo_model_instance(self):
        """A valid dotted import string returns an instantiated PyomoModel subclass."""
        instance = PyomoModel.load_from_import(_KEA_IMPORT, **_KEA_KWARGS)
        assert isinstance(instance, PyomoModel)

    def test_concrete_model_is_built(self):
        """The returned instance already has a built ConcreteModel on self.model."""
        instance = PyomoModel.load_from_import(_KEA_IMPORT, **_KEA_KWARGS)
        assert isinstance(instance.model, pyo.ConcreteModel)

    def test_kwargs_forwarded_correctly(self):
        """sampling_time and prediction_horizon are stored on the instance."""
        instance = PyomoModel.load_from_import(_KEA_IMPORT, **_KEA_KWARGS)
        assert instance.sampling_time == _KEA_KWARGS["sampling_time"]
        assert instance.prediction_horizon == _KEA_KWARGS["prediction_horizon"]

    def test_bad_module_raises_module_not_found(self):
        """A non-existent module path raises ModuleNotFoundError."""
        with pytest.raises(ModuleNotFoundError):
            PyomoModel.load_from_import("nonexistent.module.SomeClass", **_KEA_KWARGS)

    def test_bad_class_raises_attribute_error(self):
        """A valid module but non-existent class name raises AttributeError."""
        with pytest.raises(AttributeError):
            PyomoModel.load_from_import("examples.kea_tank.kea_pyomo_model.NonExistentClass", **_KEA_KWARGS)


class TestCreateState:
    """Integration tests for PyomoModel.create_state."""

    def test_creates_state_config_file(self):
        """create_state writes a state config TOML file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(_KEA_IMPORT, "kea", tmp_dir, **_KEA_KWARGS)
            assert (pathlib.Path(tmp_dir) / "kea_state_config.toml").exists()

    def test_creates_model_parameters_file(self):
        """create_state writes a model parameters TOML file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(_KEA_IMPORT, "kea", tmp_dir, **_KEA_KWARGS)
            assert (pathlib.Path(tmp_dir) / "kea_model_parameters.toml").exists()

    def test_state_config_has_indexed_vars_as_actions(self):
        """Indexed Var components (heating, temp) appear as actions in the state config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(_KEA_IMPORT, "kea", tmp_dir, **_KEA_KWARGS)
            content = (pathlib.Path(tmp_dir) / "kea_state_config.toml").read_text()
        assert "actions = [" in content
        assert 'name = "heating"' in content
        assert 'name = "temp"' in content

    def test_state_config_has_indexed_params_as_observations(self):
        """Indexed Param component (energy_price) appears as observations in the state config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(_KEA_IMPORT, "kea", tmp_dir, **_KEA_KWARGS)
            content = (pathlib.Path(tmp_dir) / "kea_state_config.toml").read_text()
        assert "observations = [" in content
        assert 'name = "energy_price"' in content

    def test_model_parameters_has_scalar_params(self):
        """Scalar Param components appear in the model_parameters TOML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(_KEA_IMPORT, "kea", tmp_dir, **_KEA_KWARGS)
            content = (pathlib.Path(tmp_dir) / "kea_model_parameters.toml").read_text()
        assert "p_heat" in content
        assert "tank_temperature_min" in content
        assert "tank_temperature_max" in content

    def test_default_output_dir_uses_cwd(self, monkeypatch, tmp_path):
        """When output_dir is None, files are created in the current working directory."""
        monkeypatch.chdir(tmp_path)
        PyomoModel.create_state(_KEA_IMPORT, "kea_cwd", None, **_KEA_KWARGS)
        assert (tmp_path / "kea_cwd_state_config.toml").exists()
        assert (tmp_path / "kea_cwd_model_parameters.toml").exists()

    def test_bootstraps_missing_model_parameters(self):
        """create_state can export files without explicitly supplied model_parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(
                _KEA_IMPORT,
                "kea_bootstrap",
                tmp_dir,
                sampling_time=10.0,
                prediction_horizon=60.0,
            )
            assert (pathlib.Path(tmp_dir) / "kea_bootstrap_state_config.toml").exists()
            assert (pathlib.Path(tmp_dir) / "kea_bootstrap_model_parameters.toml").exists()

    def test_export_path_does_not_call_model_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            PyomoModel.create_state(
                "test.test_simulators.test_pyomo_model._InitFailsModel",
                "init_bypass",
                tmp_dir,
            )
            assert (pathlib.Path(tmp_dir) / "init_bypass_state_config.toml").exists()
            assert (pathlib.Path(tmp_dir) / "init_bypass_model_parameters.toml").exists()
