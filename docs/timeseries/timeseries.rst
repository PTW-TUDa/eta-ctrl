.. _timeseries:

Timeseries
===============
Many *ETA Ctrl* functions and classes operate on timeseries data and :py:class:`pandas.DataFrame` objects
containing timeseries data. The *timeseries* module in *ETA Ctrl* provides some additional functionality for both.
It can for example find random time slices in Dataframes or import timeseries data from multiple CSV files and map
a (random if required) section of it into a Dataframe.

Scenario Data Loader
-----------------------
Scenario data is often required to perform optimizations and simulations of factory systems. The import function
can import data from multiple files and returns a cleaned Dataframe.

.. autofunction:: eta_ctrl.timeseries::scenario_from_csv
    :no-index:

Extensions for pandas.DataFrame
------------------------------------

.. automodule:: eta_ctrl.timeseries.dataframes
    :members:
    :no-index:
