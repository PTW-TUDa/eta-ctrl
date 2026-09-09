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
The :class:`~eta_ctrl.timeseries.scenario_manager.ScenarioManager` class handles provided scenario data via the config file.
It is instantiated by the :class:`~eta_ctrl.config.config.Config` class.
The environment will call the :class:`~eta_ctrl.timeseries.scenario_manager.ScenarioManager` automatically to retrieve scenario data for the current time step.

Scenario data is loaded from ``scenario_time_begin`` to ``scenario_time_end``.
With ``use_random_time_slice=False``, each episode starts at offset 0 of this interval (which is
``scenario_time_begin``). With ``use_random_time_slice=True``, a random offset is sampled only from the valid
remaining space when the loaded interval is longer than the required episode duration.
When performing MPC experiments, the loaded interval needs to be longer than the episode duration plus the prediction horizon.
In vectorized runs, each environment samples its own offset at reset,
so environments can use different scenario slices within the same global episode.

.. autoclass:: eta_ctrl.timeseries.scenario_manager::ScenarioManager
    :no-index:

Extensions for pandas.DataFrame
------------------------------------

.. automodule:: eta_ctrl.timeseries.dataframes
    :members:
    :no-index:
