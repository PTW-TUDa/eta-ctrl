from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import pyomo.environ as pyo

from eta_ctrl.envs import PyomoEnv

if TYPE_CHECKING:
    from typing import Any


log = getLogger(__name__)


class DrKea(PyomoEnv):
    version = "1.1"
    description = "Demonstration of a simple MPC environment for a tank heating system."

    def __init__(self, **kwargs: Any) -> None:
        # Instantiate PyomoEnv
        super().__init__(**kwargs)

        # Scale the fixed temperature change values from absolute seconds to relative to the sampling time
        self.model_parameters["temperature_change_heating"] *= self.sampling_time  # type: ignore[index]
        self.model_parameters["temperature_change_cleaning"] *= self.sampling_time  # type: ignore[index]

        self.use_model_time_increments = True  # Increment by one instead of the sampling time

    def _model(self) -> pyo.ConcreteModel:
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

        # Calculation of the total tank temperature
        def tank_temperature(model: pyo.ConcreteModel, t: int) -> pyo.Constraint:
            # Constraint with the initial temperature for the first time step
            if t == 0:
                return model.temp[t] == model.tank_temperature_start

            # Define temperature change based on the heating action
            # (note that we can't use an 'if' statement here)
            heating_change = model.heating[t - 1] * model.temperature_change_heating
            cleaning_change = (1 - model.heating[t - 1]) * model.temperature_change_cleaning
            temperature_change = heating_change + cleaning_change

            # Constraint for the temperature change
            return model.temp[t] == model.temp[t - 1] + temperature_change

        model.tank_temperature = pyo.Constraint(model.t, rule=tank_temperature, doc="Calcuatlion of tank temperature")

        # =============================================================================
        #     # Objective function
        # =============================================================================

        def objective_rule(model: pyo.ConcreteModel) -> pyo.Expression:
            return sum(model.heating[t] * model.energy_price[t] for t in model.t) * model.p_heat

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize, doc="Total cost of heating")

        ts = (
            self.scenario_manager.get_scenario_state_with_duration(n_step=0, duration=self.n_prediction_steps + 1)
            if self.scenario_manager is not None
            else None
        )
        return model.create_instance(data=self.pyo_component_params(component_name=None, ts=ts, index=model.t))

    ### This method does not need to be implemented
    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Reset the environment and perform the first model update
        with observations from the actual machine.

        :param seed: Seed for the random number generator.
        :param options: Options for the reset.
        :return: Initial observation and info dictionary.
        """
        super()._reset()
        return {}

    def close(self) -> None:
        """
        Perform any necessary cleanup or resource deallocation.

        This method should be implemented if the environment is holding onto
        resources such as file handles, network connections, or other external
        resources that need to be explicitly released.
        """

    def render(self) -> None:
        """Optional method to render the environment for human inspection."""
