.. _examples:

Usage examples
================
*ETA Ctrl* contains example implementations for different usages of the package.
This page gives a short overview of the examples.


ETA Ctrl Optimization
--------------------
Examples for the optimization part of the framework are also provided. The *pendulum* example is the
simplest one of them. It implements an inverse pendulum, similar to
the `equivalent example in Farama gymnasium <https://gymnasium.farama.org/environments/classic_control/pendulum/>`_.
The environment can be used for
different kinds of agents and includes examples for the PPO reinforcement learning
agent as well as a simple rule based controller.

The *damped_oscillator* example illustrates how simulation environments are created,
based on the *BaseEnvSim* class. In this simple example, only the StateConfig and the
render function need to be specified to obtain a completely functional environment.
In the example, the controller will just supply random action values.

Finally, the *cyber_physical_system* example shows the full capabilities of the *ETA Ctrl*
framework. It utilizes the interaction between a simulation and an actual machine to
supply advanced observations to an agent which controls the tank heating unit of
an industrial parts cleaning machine. To be able to run this example, a Dymola license is needed.
In the :ref:`dymola_license_not_found` it is explained how to use the license.
