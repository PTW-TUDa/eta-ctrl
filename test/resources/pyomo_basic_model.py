from __future__ import annotations

import pyomo.environ as pyo

from eta_ctrl.simulators.pyomo_model import PyomoModel


class PyomoBasicModel(PyomoModel):
    """Tank temperature model for testing MpcAgent, structured after DrKeaModel.

    x[t] is a binary heating decision. temp[t] is the tank temperature.
    Dynamics: temp[t] = temp[t-1] + (x[t-1] - 0.5) * 4
        → heating (x=1): +2°C, cooling (x=0): -2°C.
    Constraint: temp[t] >= temp_min (default 50°C).
    Default prices p[t]=1 except p[1]=10 (step 1 is expensive).
    Objective: minimize sum(x[t] * p[t]).

    With temp0=50 (at minimum): must heat immediately → x[0]=1.
    With temp0=60 (headroom): can defer heating → x[0]=0.
    """

    _start_value_mapping = {"temp0": "temp_expression"}

    def _model(self) -> pyo.AbstractModel:
        # Default prices: cheap (1) at all steps except step 1 which is expensive (10)
        prices = {t: 1 if t != 1 else 10 for t in range(self.n_prediction_steps + 1)}

        model = pyo.AbstractModel()
        model.T = pyo.RangeSet(0, self.n_prediction_steps)

        model.temp0 = pyo.Param(initialize=55, mutable=True)
        model.temp_min = pyo.Param(initialize=50, mutable=True)
        model.p = pyo.Param(model.T, initialize=prices, mutable=True)

        model.temp = pyo.Var(model.T, within=pyo.Reals, bounds=(model.temp_min, None))
        model.x = pyo.Var(model.T, within=pyo.Binary)

        # === Expression === #
        def temp_expression(m: pyo.ConcreteModel, t: int) -> float:
            if t == 0:
                return m.temp0
            temp_change = (m.x[t - 1] - 0.5) * 4  # either -2 or 2
            return m.temp_expression[t - 1] + temp_change

        model.temp_expression = pyo.Expression(model.T, rule=temp_expression)

        # === Constraint === #
        def temp_constraint(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return m.temp[t] == m.temp_expression[t]

        model.temp_constraint = pyo.Constraint(model.T, rule=temp_constraint)

        # === Objective === #
        def objective_rule(m):
            return sum(m.x[t] * m.p[t] for t in m.T)

        model.obj = pyo.Objective(rule=objective_rule)

        return model
