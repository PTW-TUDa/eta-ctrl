from __future__ import annotations

import re

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


@pytest.fixture(scope="module")
def model():
    return _SimpleModel(sampling_time=10, prediction_horizon=60)


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
        solution = model.pyo_get_solution()
        assert solution["indexed_param"] == self.indexed_values
        assert solution["scalar_param"] == self.scalar_value
