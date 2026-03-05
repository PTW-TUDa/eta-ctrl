from __future__ import annotations

import pyomo.environ as pyo

from eta_ctrl.simulators.pyomo_model import PyomoModel


class MPCBasicModel(PyomoModel):
    """PyomoModel for testing MpcAgent. Implements a simple quadratic optimization problem.

    Minimizes (x[0] - 1)² + (u[0] + 1)².
    """

    def _model(self) -> pyo.AbstractModel:
        model = pyo.AbstractModel()

        model.T = pyo.RangeSet(0, self.n_prediction_steps)

        model.x0 = pyo.Param(initialize=0, mutable=True)
        model.x = pyo.Var(model.T, initialize=0)
        model.u = pyo.Var(model.T, bounds=(-5, 5), initialize=0)

        def obj_rule(m: pyo.ConcreteModel) -> pyo.Expression:
            return sum((m.x[t] - 1) ** 2 + (m.u[t] + 1) ** 2 for t in m.T)

        model.obj = pyo.Objective(rule=obj_rule)

        def constr_rule(m: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            if t == 0:
                return m.x[0] == m.x0
            return m.x[t] == m.x[t - 1] + m.u[t - 1]

        model.constr = pyo.Constraint(model.T, rule=constr_rule)

        return model
