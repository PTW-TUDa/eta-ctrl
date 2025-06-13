ETA Ctrl Framework
######################

The `ETA Ctrl` framework provides a standardized interface for developing digital twins of factories or machines in a factory. It is designed to facilitate rolling horizon optimization, simulation, and interaction with factory systems. The framework is based on the Gymnasium environment and integrates seamlessly with tools like FMUs, Julia, Pyomo models, and live connections to real-world assets.

Documentation
*****************

Full Documentation can be found on the
`Documentation Page <https://eta-ctrl.readthedocs.io/>`_. (TODO: Make this a link as soon as first docs version is released.)

.. warning::
    This is beta software. APIs and functionality might change without prior notice. Please fix the version you
    are using in your requirements to ensure your software will not be broken by changes in *ETA Ctrl*.

Utilities Overview
********************

Optimization Utilities
==========================

- **`ETA Ctrl`**: Central controller for managing optimization workflows, including learning and execution processes.

Configuration Utilities
==========================

- **`ConfigOpt`**: Represents the configuration for an optimization run.
- **`ConfigOptSetup`**: Defines setup configurations for optimization runs.
- **`ConfigOptSettings`**: Represents settings for optimization runs.
- **`ConfigOptRun`**: Handles paths and metadata for optimization runs.

Environment Utilities
==========================

- **Base Classes**:

  - **`BaseEnv`**: Abstract base class for creating custom environments.
  - **`BaseEnvLive`**: Extends `BaseEnv` for live environments interacting with real-world systems.
  - **`BaseEnvMPC`**: Extends `BaseEnv` for environments using Model Predictive Control (MPC).
  - **`BaseEnvSim`**: Extends `BaseEnv` for environments using FMU-based simulations.
  - **`JuliaEnv`**: Environment class for interacting with Julia-based simulation models.

- **Vectorization**:

  - **`NoVecEnv`**: Custom vectorizer for environments that handle multithreading internally.

Simulation Utilities
==========================

- **`FMUSimulator`**: Provides functionality for simulating FMUs (Functional Mock-up Units).

Time Series Utilities
==========================

- **`scenario_from_csv`**: Imports and processes scenario data from CSV files.
- **`df_from_csv`**: Reads time series data from a CSV file and returns it as a pandas DataFrame.
- **`df_resample`**: Resamples the time index of a DataFrame to a specified frequency.
- **`df_interpolate`**: Interpolates missing values in a DataFrame with a specified frequency.

State Management Utilities
==========================

- **`StateVar`**: Represents a single variable in the state of an environment.
- **`StateConfig`**: Configures the action and observation spaces based on `StateVar` instances.

Contributing
*****************

Please read the `development guide <https://eta-utility.readthedocs.io/en/main/guide/development.html>`_ before starting development on *ETA Ctrl*


Citing this Project / Authors
******************************

See `AUTHORS.rst` for a full list of contributors.

Please cite this repository as:

  .. code-block::

    Grosch, B., Ranzau, H., Dietrich, B., Kohne, T., Fuhrländer-Völker, D., Sossenheimer, J., Lindner, M., Weigold, M.
    A framework for researching energy optimization of factory operations.
    Energy Inform 5 (Suppl 1), 29 (2022). https://doi.org/10.1186/s42162-022-00207-6
