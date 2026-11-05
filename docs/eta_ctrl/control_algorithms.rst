.. _agents:

Control Algorithms
=======================
The agents implemented in *eta_ctrl.agents* are subclasses of
:py:class:`stable_baselines3.common.base_class.BaseAlgorithm` in *stable_baselines3*. Calling them agents
is a remnant from *stable_baselines2* (the wording was changed in *stable_baselines3*).

Usually there is no need to dive more deeply into the agents provided by *ETA Ctrl*. You can use them by specifying
their import path in your experiment configuration and don't have to worry about how they work. It is good to know
however, that some agents do not implement all methods which would be required by the interface in normal usage. Within the *ETA Ctrl* framework this usually isn't a problem, because the methods are not used.

The currently available agents are listed here. Note that you need to specify the parameters required for
instantiation in the *agent_specific* section of the *ETA Ctrl* configuration file.

Non-learning agents
---------------------------------------
Agents which do not learn, such as the MPC agent, are subclasses of
:py:class:`~eta_ctrl.agents.non_learning_agent.NonLearningAgent`. They determine their actions from the observations
by an explicit decision logic (e.g. an optimization), so there is no policy network and no learned state.

.. warning::

    Non-learning agents are only partially compatible with *stable_baselines3*. They inherit from *BaseAlgorithm* so
    that they can be used by :py:class:`~eta_ctrl.core.EtaCtrl`, but only *predict* is functional:

    - :py:meth:`EtaCtrl.learn() <eta_ctrl.core.EtaCtrl.learn>` logs an error and does nothing. Use
      :py:meth:`EtaCtrl.play() <eta_ctrl.core.EtaCtrl.play>` instead.
    - *save()* and *load()* raise a *NotImplementedError*.
    - Do not rely on other *stable_baselines3* features (policy, parameters, callbacks, ...).

Non-learning agents identify the actions by name. The order of the action array is defined by the actions of the
*StateConfig* and is passed to the agent as the *actions_order* parameter. This is done automatically by
:py:class:`~eta_ctrl.config.config.Config`; do not set it in the *agent_specific* section.

.. autoclass:: eta_ctrl.agents::NonLearningAgent
    :no-index:

MPC Agent
---------------------------------------
The MpcAgent agent implements a model predictive controller. It can be used to solve mathematical models in conjunction
with mathematical solvers such as gurobi, cplex or glpk and it relies on the *pyomo* library to achieve this.

Solver options are specified with the *solver_options* parameter, which is passed on to the solver.

.. autoclass:: eta_ctrl.agents::MpcAgent
    :no-index:

Rule Based Agent (Base Class)
---------------------------------------
The rule based agent is a base class which facilitates the creation of simple rule based agents. To use it, you need
to implement the :py:meth:`eta_ctrl.agents.RuleBased.control_rules` method. The *control_rules* method
takes the array of observations from the environment and determines an array of actions based on them.

.. autoclass:: eta_ctrl.agents::RuleBased
    :members: control_rules
    :no-index:
