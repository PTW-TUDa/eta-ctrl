"""Pyomo ConcreteModel fixtures for testing export functionality.

This module provides pytest fixtures that create ConcreteModel instances for comprehensive
testing of Pyomo model export functionality, including state configuration and parameter exports.
"""

from __future__ import annotations

import logging

import pyomo.environ as pyo
import pytest

from eta_ctrl.envs.pyomo_env import PyomoEnv

# Set up logging for the test module
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def production_planning_concrete_model() -> pyo.ConcreteModel:
    """Create a production planning ConcreteModel fixture for Pyomo export testing.

    This pytest fixture creates a ConcreteModel representing a multi-period, multi-product
    production planning problem with capacity constraints, demand requirements, and cost
    optimization. The ConcreteModel includes various types of variables (scalar, indexed,
    binary) and parameters to comprehensively test all aspects of the Pyomo export functionality.

    Returns:
        ConcreteModel: A Pyomo ConcreteModel fixture for production planning export testing.
    """
    model = pyo.ConcreteModel("ProductionPlanningModel")

    # Sets for indexed components
    model.time_periods = pyo.Set(initialize=[1, 2, 3, 4], doc="Planning time periods")
    model.products = pyo.Set(initialize=["A", "B", "C"], doc="Product types")
    model.machines = pyo.Set(initialize=["M1", "M2"], doc="Available machines")

    # Scalar parameters
    model.total_capacity = pyo.Param(initialize=500.0, doc="Total production capacity")
    model.base_cost = pyo.Param(initialize=10.0, doc="Base production cost per unit")
    model.inventory_cost = pyo.Param(initialize=2.0, doc="Inventory holding cost per unit")
    model.overtime_rate = pyo.Param(initialize=1.5, doc="Overtime cost multiplier")

    # Indexed parameters - Demand by time period
    model.period_demand = pyo.Param(
        model.time_periods, initialize={1: 80, 2: 120, 3: 100, 4: 140}, doc="Demand for each time period"
    )

    # Indexed parameters - Production cost by product
    model.product_cost = pyo.Param(
        model.products,
        initialize={"A": 15.0, "B": 25.0, "C": 35.0},
        doc="Production cost per unit for each product",
    )

    # Indexed parameters - Machine capacity by time period
    model.machine_capacity = pyo.Param(
        model.machines,
        model.time_periods,
        initialize={
            ("M1", 1): 60,
            ("M1", 2): 70,
            ("M1", 3): 65,
            ("M1", 4): 75,
            ("M2", 1): 50,
            ("M2", 2): 60,
            ("M2", 3): 55,
            ("M2", 4): 65,
        },
        doc="Capacity of each machine in each time period",
    )

    # Indexed parameters - Processing time by product and machine
    model.processing_time = pyo.Param(
        model.products,
        model.machines,
        initialize={
            ("A", "M1"): 1.2,
            ("A", "M2"): 1.5,
            ("B", "M1"): 2.0,
            ("B", "M2"): 1.8,
            ("C", "M1"): 2.5,
            ("C", "M2"): 2.2,
        },
        doc="Processing time per unit for each product-machine combination",
    )

    # Scalar decision variables
    model.total_production = pyo.Var(bounds=(0, None), doc="Total production across all periods")
    model.total_inventory = pyo.Var(bounds=(0, 200), doc="Total inventory held")
    model.total_cost = pyo.Var(bounds=(0, None), doc="Total production cost")
    model.overtime_hours = pyo.Var(bounds=(0, 100), doc="Total overtime hours used")

    # Indexed decision variables - Production by time period
    model.period_production = pyo.Var(
        model.time_periods, bounds=(0, 150), doc="Production quantity in each time period"
    )

    # Indexed decision variables - Production by product
    model.product_production = pyo.Var(
        model.products, bounds=(0, 200), domain=pyo.NonNegativeReals, doc="Production quantity for each product"
    )

    # Indexed decision variables - Inventory by product and time period
    model.inventory = pyo.Var(
        model.products,
        model.time_periods,
        bounds=(0, 50),
        doc="Inventory level for each product in each time period",
    )

    # Indexed decision variables - Machine usage
    model.machine_usage = pyo.Var(
        model.machines,
        model.time_periods,
        bounds=(0, None),
        doc="Hours of usage for each machine in each time period",
    )

    # Binary decision variables - Machine activation
    model.machine_active = pyo.Var(
        model.machines,
        model.time_periods,
        domain=pyo.Binary,
        doc="Whether each machine is active in each time period",
    )

    # Scalar constraints
    model.total_capacity_constraint = pyo.Constraint(
        expr=model.total_production <= model.total_capacity, doc="Total production cannot exceed capacity"
    )

    model.total_cost_definition = pyo.Constraint(
        expr=model.total_cost
        == model.total_production * model.base_cost + model.total_inventory * model.inventory_cost,
        doc="Definition of total cost",
    )

    # Indexed constraints - Period capacity constraints
    def period_capacity_rule(model, t):
        return model.period_production[t] <= 120

    model.period_capacity_constraint = pyo.Constraint(
        model.time_periods, rule=period_capacity_rule, doc="Production capacity limit for each period"
    )

    # Indexed constraints - Machine capacity constraints
    def machine_capacity_rule(model, m, t):
        return model.machine_usage[m, t] <= model.machine_capacity[m, t]

    model.machine_capacity_constraint = pyo.Constraint(
        model.machines, model.time_periods, rule=machine_capacity_rule, doc="Machine usage cannot exceed capacity"
    )

    # Indexed constraints - Production balance
    def production_balance_rule(model, p):
        return sum(model.inventory[p, t] for t in model.time_periods) >= 10

    model.production_balance_constraint = pyo.Constraint(
        model.products, rule=production_balance_rule, doc="Minimum inventory requirements for each product"
    )

    # Linking constraint - Total production
    model.total_production_link = pyo.Constraint(
        expr=model.total_production == sum(model.period_production[t] for t in model.time_periods),
        doc="Link total production to period productions",
    )

    # Objective function - Minimize total cost
    model.minimize_cost = pyo.Objective(
        expr=model.total_cost + model.overtime_hours * model.overtime_rate * model.base_cost,
        sense=pyo.minimize,
        doc="Minimize total production and overtime costs",
    )

    return model


if __name__ == "__main__":
    # Example usage for testing ConcreteModel export functionality

    # Create the production planning ConcreteModel fixture
    prod_model = production_planning_concrete_model()
    log.info(
        f"Created production planning ConcreteModel with {len(list(prod_model.component_objects(pyo.Var)))} variables"
    )
    log.info(f"ConcreteModel has {len(list(prod_model.component_objects(pyo.Param)))} parameters")

    # Test create_state functionality with the ConcreteModel fixture
    PyomoEnv.create_state(prod_model, "production_planning", ".")
    log.info("✓ Successfully created state files for production planning ConcreteModel")
