MPC/Pyomo Migration Guide
=========================

Summary
-------

MPC experiments were redesigned to separate model logic from environment logic.
The previous ``MathSolver`` + ``PyomoEnv`` setup is now built around ``MpcAgent`` + ``PyomoModel``.

**What changed:**

- ``MathSolver`` was replaced by ``MpcAgent``
- MPC model definition moved from ``PyomoEnv._model()`` into a dedicated ``PyomoModel`` subclass
- ``prediction_horizon`` is configured in ``[settings]``
- ``model_parameters`` are configured in ``[settings.agent.model_parameters]``
- ``MpcAgent`` now requires ``model_import`` (dotted import path to your ``PyomoModel``)

**Estimated effort:** 5-10 minutes for MPC experiments.


Config File Changes
-------------------

1. Replace ``MathSolver`` with ``MpcAgent``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the agent class in ``[setup]``:

.. literalinclude:: /../examples/kea_tank/config.toml
    :language: toml
    :start-after: --migration-agent-setup-start--
    :end-before: --migration-agent-setup-end--

2. Keep ``prediction_horizon`` in ``[settings]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``prediction_horizon`` is now a framework-level setting, validated together with ``sampling_time``.

.. literalinclude:: /../examples/kea_tank/config.toml
    :language: toml
    :start-after: --migration-prediction-horizon-start--
    :end-before: --migration-prediction-horizon-end--

3. Move model settings under ``[settings.agent]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define ``model_import`` and put all model parameters under ``[settings.agent.model_parameters]``:

Solver-specific options are now passed via a dedicated ``solver_options`` mapping
instead of defining solver keys directly in ``[settings.agent]``.

**Before:**

.. code-block:: toml

    [settings.agent]
    model_import = "my_package.my_model.MyModel"
    solver_name = "cplex_direct"
    timelimit = 30

**After:**

.. code-block:: toml

    [settings.agent]
    model_import = "my_package.my_model.MyModel"
    solver_name = "cplex_direct"
    [settings.agent.solver_options]
    timelimit = 30
    # or solver_options = {timelimit = 30}

*Full config after migration:*

.. literalinclude:: /../examples/kea_tank/config.toml
    :language: toml
    :start-after: --migration-agent-config-start--
    :end-before: --migration-agent-config-end--


Python Code Changes
-------------------

4. Move ``_model()`` from ``PyomoEnv`` to ``PyomoModel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimization model definition stays mostly unchanged, but it now lives in a ``PyomoModel`` subclass.

.. .. literalinclude:: /../examples/kea_tank/kea_pyomo_model.py
..     :language: python
..     :start-after: --migration-model-method-start--
..     :end-before: --migration-model-method-end--

5. Feed current state values as observations in the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this architecture, the environment provides current values that are used by the model parameters.

.. literalinclude:: /../examples/kea_tank/environments/kea_model_env.py
    :language: python
    :start-after: --migration-env-observation-start--
    :end-before: --migration-env-observation-end--


Optional New Feature: PyomoSimEnv
---------------------------------

``PyomoSimEnv`` is a new environment type and not a required migration step.
Use it when you want to run a Pyomo model as a simulation model for the next timestep,
without MPC optimization and without calling a solver.

In this mode, the model is evaluated step-by-step using mapped expressions:

- Define ``model_import`` to a ``PyomoModel`` subclass in your environment
- Provide ``_start_value_mapping`` in the ``PyomoModel``
- Use external outputs/state values as handover parameters for the next step

If you only migrate existing MPC experiments (``MpcAgent`` + optimization), you can skip this section.


Migration Checklist
-------------------

- Replace ``agent_import = "eta_ctrl.agents.MathSolver"`` with ``"eta_ctrl.agents.MpcAgent"``
- Create a ``PyomoModel`` class and move your ``_model()`` implementation there
- Set ``model_import`` in ``[settings.agent]``
- Keep ``prediction_horizon`` in ``[settings]``
- Move model parameters to ``[settings.agent.model_parameters]``


Common Issues
-------------

Missing ``model_import``
^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** Agent initialization fails when creating the MPC model.

**Fix:** Add ``model_import`` under ``[settings.agent]`` and point it to your ``PyomoModel`` subclass.


Invalid prediction horizon setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptom:** ``prediction_horizon`` is not found.

**Fix:** Ensure ``prediction_horizon`` is placed in ``[settings]`` and not in ``[settings.agent]``.
