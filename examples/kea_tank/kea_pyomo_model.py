from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import pyomo.environ as pyo

from eta_ctrl.simulators.pyomo_model import PyomoModel

if TYPE_CHECKING:
    from typing import Any


log = getLogger(__name__)


class DrKeaModel(PyomoModel):
    def __init__(self, sampling_time: float, model_parameters: dict[str, Any], **kwargs: Any) -> None:
        self._start_value_mapping = {"tank_temperature_start": "temp_expr"}

        # Scale the fixed temperature change values from absolute seconds to relative to the sampling time
        model_parameters["temperature_change_heating"] *= sampling_time
        model_parameters["temperature_change_cleaning"] *= sampling_time

        # Instantiate PyomoModel
        super().__init__(sampling_time=sampling_time, model_parameters=model_parameters, **kwargs)

        self._use_model_time_increments = True  # Increment by one instead of the sampling time

    def _model(self) -> pyo.AbstractModel:
        """This is where the actual model is defined.

        :return: The Pyomo model.
        """
        # =============================================================================
        #     #Model definition
        # =============================================================================

        model = pyo.AbstractModel()

        # =============================================================================
        #     # Model parameters and sets
        # =============================================================================

        model.t = pyo.RangeSet(0, self.n_prediction_steps, doc="Index list of discrete time steps")

        # Tank temperature constants (as defined in the config file)
        model.p_heat = pyo.Param(within=pyo.Reals, doc="Power consumption of heating")

        model.tank_temperature_start = pyo.Param(within=pyo.Reals, mutable=True, doc="Tank temperature for k = 0")

        model.tank_temperature_min = pyo.Param(within=pyo.Reals, doc="Lower limit for tank temperature")
        model.tank_temperature_max = pyo.Param(within=pyo.Reals, doc="Upper limit for tank temperature")

        model.temperature_change_heating = pyo.Param(
            within=pyo.Reals, doc="Constant tank temperature increase of tank heater"
        )
        model.temperature_change_cleaning = pyo.Param(
            within=pyo.Reals, doc="Constant tank temperature drop during cleaning"
        )

        # Energyprices (from external data)
        model.energy_price = pyo.Param(model.t, mutable=True, doc="List of energy prices for all time steps")

        # =============================================================================
        #     # Model variables
        # =============================================================================

        # Heating boolean variable, controlled by the agent
        model.heating = pyo.Var(model.t, within=pyo.Binary, doc="Is true if heater is on")

        # Tank temperature variable, not controlled by the agent
        model.temp = pyo.Var(
            model.t,
            within=pyo.Reals,
            bounds=(model.tank_temperature_min, model.tank_temperature_max),
            doc="Tank temperature",
        )

        # =============================================================================
        #     # Model constraints
        # =============================================================================

        def temp_change_logic(model: pyo.ConcreteModel, t: int) -> float:
            # Constraint with the initial temperature for the first time step
            if t == 0:
                return model.tank_temperature_start

            is_heating = model.heating[t - 1]
            heating_change = is_heating * model.temperature_change_heating
            cleaning_change = (1 - is_heating) * model.temperature_change_cleaning

            return model.temp_expr[t - 1] + heating_change + cleaning_change

        model.temp_expr = pyo.Expression(model.t, rule=temp_change_logic, doc="Calculation of tank temperature")

        # Calculation of the total tank temperature
        def tank_temperature_constraint(model: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            return model.temp[t] == model.temp_expr[t]

        model.tank_temperature = pyo.Constraint(
            model.t, rule=tank_temperature_constraint, doc="Calcuatlion of tank temperature"
        )

        # =============================================================================
        #     # Objective function
        # =============================================================================

        def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
            return sum(model.heating[t] * model.energy_price[t] for t in model.t) * model.p_heat

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc="Total cost of heating")

        return model
