Config Migration Guide
======================

Summary
-------

The configuration system has been rewritten from attrs to Pydantic v2. This simplifies the internal
class structure (removing manual ``from_dict`` parsing, converter chains, and the :py:class:`~eta_ctrl.config.config_run.ConfigRun` coupling)
and brings automatic type validation, better error messages, and JSON schema generation for editor
autocompletion.

**Design principle:** The ``[settings]`` section now contains all framework-level parameters that are
validated and documented (e.g. ``sampling_time``, ``prediction_horizon``, ``scenario_files``). The
``[settings.environment]`` and ``[settings.agent]`` subsections are reserved for values specific to the
chosen environment or agent subclass — these are passed through without framework-level validation.

**What changed for config files:**

- ``[environment_specific]`` and ``[agent_specific]`` are now nested under ``[settings]``
- ``prediction_horizon``, ``scenario_files``, ``scenario_time_begin``, and ``scenario_time_end`` moved from
  the environment section to ``[settings]``
- The ``_package`` + ``_class`` import syntax in ``[setup]`` is removed; use ``_import`` only

**What changed for Python code:**
Config creation through :py:class:`~eta_ctrl.core.EtaCtrl` works as before — no changes needed.
If you created :py:class:`~eta_ctrl.config.config.Config` objects manually in custom code, refer to the updated API documentation.

**Estimated effort:** Most config files need 5-10 minutes of restructuring.


Config File Changes
-------------------

1. Rename ``[environment_specific]`` and ``[agent_specific]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These sections are now nested under ``[settings]``:

**Before:**

.. code-block:: toml

   [settings]
   sampling_time = 10
   episode_duration = 600

   [environment_specific]
   sim_steps_per_sample = 1

   [agent_specific]
   action_index = 1
   solver_name = "cplex_direct"

**After:**

.. code-block:: toml

   [settings]
   sampling_time = 10
   episode_duration = 600

   [settings.environment]
   sim_steps_per_sample = 1

   [settings.agent]
   action_index = 1
   solver_name = "cplex_direct"

The old names ``environment_specific`` / ``env_specific`` and ``agent_specific`` still work as aliases,
but the new canonical names are ``environment`` and ``agent``.

2. Move ``prediction_horizon`` to ``[settings]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``prediction_horizon`` is now a top-level settings parameter, not an environment-specific one.
It is validated against ``sampling_time`` at load time.

**Before:**

.. code-block:: toml

   [environment_specific]
   prediction_horizon = 1200
   sim_steps_per_sample = 1

**After:**

.. code-block:: toml

   [settings]
   prediction_horizon = 1200

   [settings.environment]
   sim_steps_per_sample = 1

3. Move ``scenario_files`` to ``[settings]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scenario_files`` is now declared directly under ``[settings]``, not inside the environment section.

**Before:**

.. code-block:: toml

   [[environment_specific.scenario_files]]
   path = "electricity_price_test.csv"
   interpolation_method = "ffill"

**After:**

.. code-block:: toml

   [settings]
   scenario_files = [
      {path = "electricity_price_test.csv", interpolation_method = "ffill"}
   ]

4. Move ``scenario_time_begin`` and ``scenario_time_end`` to ``[settings]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These were previously in the environment section and are now top-level settings parameters.

**Before:**

.. code-block:: toml

   [environment_specific]
   scenario_time_begin = "2026-03-18 00:00"
   scenario_time_end = "2026-03-18 00:50"

**After:**

.. code-block:: toml

   [settings]
   scenario_time_begin = "2026-03-18 00:00"
   scenario_time_end = "2026-03-18 00:50"

5. Use ``_import`` syntax only in ``[setup]``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The old ``_package`` + ``_class`` combo syntax is removed. Use the full dotted import path with
``_import``.

**Before:**

.. code-block:: toml

   [setup]
   environment_import = "eta_ctrl.envs.PyomoEnv"
   agent_import = "eta_ctrl.agents.MpcAgent"
   vectorizer_class = "DummyVecEnv"
   vectorizer_package = "stable_baselines3.common.vec_env.dummy_vec_env"
   policy_class = "NoPolicy"
   policy_package = "eta_ctrl.common"

**After:**

.. code-block:: toml

   [setup]
   environment_import = "eta_ctrl.envs.PyomoEnv"
   agent_import = "eta_ctrl.agents.MpcAgent"
   vectorizer_import = "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"
   policy_import = "eta_ctrl.common.NoPolicy"


Complete Before/After Example
-----------------------------

**Before (old format):**

.. code-block:: toml

   [setup]
   environment_import = "eta_ctrl.envs.PyomoEnv"
   agent_import = "eta_ctrl.agents.MpcAgent"
   vectorizer_class = "DummyVecEnv"
   vectorizer_package = "stable_baselines3.common.vec_env.dummy_vec_env"
   policy_class = "NoPolicy"
   policy_package = "eta_ctrl.common"

   [paths]
   results_relpath = "results"
   scenarios_relpath = "scenarios"

   [settings]
   sampling_time = 10
   episode_duration = 600
   n_episodes_play = 1
   verbose = 2
   seed = 123

   [environment_specific]
   sim_steps_per_sample = 1
   prediction_horizon = 1200
   scenario_time_begin = "2026-03-18 00:00"
   scenario_time_end = "2026-03-18 00:50"
   scenario_files = [
      {path = "electricity_price_test.csv", interpolation_method = "ffill"}
   ]

   [environment_specific.model_parameters]
   N = 5
   n_start = 1

   [agent_specific]
   action_index = 1
   solver_name = "cplex_direct"

**After (new format):**

.. literalinclude:: /../examples/config/config_full.toml
   :language: toml

Minimal Config Example
------------------------

A minimal config which only defines required values looks like this:

.. literalinclude:: /../examples/config/config_minimal.toml
   :language: toml

Including scenario data (expecting scenario files in the default directory ``scenarios/``):

.. literalinclude:: /../examples/config/config_minimal_with_scenarios.toml
   :language: toml

Validation Improvements
-----------------------

The new config system validates types and constraints at load time. Invalid configs that previously
silently produced wrong behavior will now raise a ``pydantic.ValidationError`` with a clear message.

Examples of values that are now caught automatically:

.. list-table::
   :header-rows: 1

   * - Field
     - Constraint
   * - ``sampling_time``
     - Must be > 0
   * - ``episode_duration``
     - Must be > 0, rounded down to multiple of ``sampling_time``
   * - ``prediction_horizon``
     - Must be > 0 (if set), rounded down to multiple of ``sampling_time``
   * - ``n_environments``
     - Must be >= 1
   * - ``verbose``
     - Must be 0-3
   * - ``setup.*_import``
     - Must resolve to a valid subclass of the expected base class

JSON Schema
-----------

A JSON schema is now generated from the Pydantic models and can be used for editor autocompletion and validation.
Use the provided files in ``.vscode/`` to activate automatic schema validation for VSCode (is also included in the ETA Ctrl Experiment Template).
