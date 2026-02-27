.. _timeseries:

Timeseries
===============
Many *ETA Ctrl* functions and classes operate on timeseries data and :py:class:`pandas.DataFrame` objects
containing timeseries data. The *timeseries* module in *ETA Ctrl* provides some additional functionality for both.
It can for example find random time slices in Dataframes or import timeseries data from multiple CSV files and map
a (random if required) section of it into a Dataframe.

ScenarioManager
-----------------------
Scenario data is often required to perform optimizations and simulations of factory systems.
The :class:`~eta_ctrl.scenarios.scenario_manager.ScenarioManager` class handles provided scenario data via the config file.
It is instantiated by the :class:`~eta_ctrl.config.config.Config` class.

When the `use_random_time_slice` argument is set to `True`

.. autoclass:: eta_ctrl.timeseries.scenario_manager::ScenarioManager
    :no-index:

Extensions for pandas.DataFrame
------------------------------------

.. automodule:: eta_ctrl.timeseries.dataframes
    :members:
    :no-index:
