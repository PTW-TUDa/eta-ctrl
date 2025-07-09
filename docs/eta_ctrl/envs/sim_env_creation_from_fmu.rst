Creating a Simulation Environment from an existing Functional Mockup Unit
===========

This Chapter describes how you can quickly generate an environment (SimEnv) from your existing .fmu(Functional Mockup Unit) file.
This process aims to make it easier to create, configure, and validate environments based on FMU simulations.

Creating an Environment from FMU
-------------------------------
The most powerful feature is the ability to create an environment directly from an FMU file using the ``SimEnv.from_fmu()`` class method.
This method handles all the necessary setup with minimal configuration:

Assuming you have already created an experiment repository, let's call it ``heater-control``. And you have installed it with poetry.

1. Put ``heating_tank.fmu`` into the main folder ``heater_control`` or a sub-folder of it, e.g. ``heater_control/environments``.
2. Run ``poetry run create_sim_env heater_control/heating_tank.fmu``
    Alternatively, in a Python console ``poetry run python``, run

.. code-block:: python
    from eta_ctrl.envs.sim_env import SimEnv

    SimEnv.from_fmu("heater_control/heating_tank.fmu")

This will:
- Create ``heating_tank.py``, containing ``HeatingTankEnv``, a subclass of ``SimEnv``.
- Create ``heating_tank_structure.toml``, containing the top-level inputs and outputs, as well as parameters of the fmu.

You still need to:
- Adjust the bounds of state variables (low_value/high_value)
- Add non-top-level inputs and outputs, if needed
- Check model parameter values

.. note::
   In the generation process, variable names from the FMU are used directly in the state configuration **without any transformation or relative naming**.
   For any state variable that interacts with a FMU the ``name`` property of a StateVar is the same as the FMU variable name.
   If you want to modify the StateVar ``name``, simply define the ``ext_id`` property and set it to the FMU variable name.

Exporting just the FMU Structure
----------------------

If you already have an Environment, you can also just export the structure of an FMU file to a TOML file using ``SimEnv.export_fmu_structure()``.
This method extracts the inputs, outputs, and parameters from the FMU to a TOML file.

.. code-block:: bash
    poetry run export_fmu_structure heater_control/heating_tank.fmu


This will create ``heating_tank_structure.toml``, containing the top-level inputs and outputs, as well as parameters of the fmu.
This will not overwrite an existing toml file, but create a new one with another name.

The generated TOML file includes:
- Basic FMU information (name, path, description)
- Actions (FMU input variables) with bounds, if set in the FMU
- Observations (FMU output variables) with bounds, if set in the FMU
- Model Parameters with their default value
