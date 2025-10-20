ETA Ctrl Framework
######################

The `ETA Ctrl` framework provides a standardized interface for developing digital twins of factories or machines in a factory.
It is designed to facilitate rolling horizon optimization, simulation, and interaction with factory systems.
The framework is based on the Gymnasium environment and integrates seamlessly with tools like FMUs, Julia, Pyomo models, and live connections to real-world assets.

Documentation
*****************

Full Documentation can be found on the `Documentation Page <https://eta-ctrl.readthedocs.io/>`_.

.. warning::
    This is beta software. APIs and functionality might change without prior notice. Please fix the version you
    are using in your requirements to ensure your software will not be broken by changes in *ETA Ctrl*.

Overview
********************

Core
==========================

- **`EtaCtrl`**: Central controller for managing optimization workflows, including learning and execution processes.

Configuration
==========================

- **`Config`**: Represents the configuration for an optimization run.
- **`RunInfo`**: Handles paths and metadata for optimization runs.

Environment
==========================

- **Base Classes**:

  - **`BaseEnv`**: Abstract base class for creating custom environments.
  - **`LiveEnv`**: Extends `BaseEnv` for live environments interacting with real-world systems.
  - **`PyomoEnv`**: Extends `BaseEnv` for environments using Model Predictive Control (MPC).
  - **`SimEnv`**: Extends `BaseEnv` for environments using FMU-based simulations.
  - **`JuliaEnv`**: Environment class for interacting with Julia-based simulation models.

- **Vectorization**:

  - **`NoVecEnv`**: Custom vectorizer for environments that handle multithreading internally.

Simulation
==========================

- **`FMUSimulator`**: Provides functionality for simulating FMUs (Functional Mock-up Units).

Time Series
==========================

- **`scenario_from_csv`**: Imports and processes scenario data from CSV files.
- **`df_from_csv`**: Reads time series data from a CSV file and returns it as a pandas DataFrame.
- **`df_resample`**: Resamples the time index of a DataFrame to a specified frequency.
- **`df_interpolate`**: Interpolates missing values in a DataFrame with a specified frequency.

State Management
==========================

- **`StateVar`**: Represents a single variable in the state of an environment.
- **`StateConfig`**: Configures the action and observation spaces based on `StateVar` instances.

Contributing
*****************

Please read the `development guide <https://eta-ctrl.readthedocs.io/en/latest/guide/development.html>`_ before starting development on *ETA Ctrl*


Citing this Project / Authors
******************************

See `AUTHORS.rst` for a full list of contributors.

Please cite this repository as:

  .. code-block::

    Grosch, B., Ranzau, H., Dietrich, B., Kohne, T., Fuhrländer-Völker, D., Sossenheimer, J., Lindner, M., Weigold, M.
    A framework for researching energy optimization of factory operations.
    Energy Inform 5 (Suppl 1), 29 (2022). https://doi.org/10.1186/s42162-022-00207-6
