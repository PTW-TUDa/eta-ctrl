from __future__ import annotations

import inspect
import pathlib
from typing import TYPE_CHECKING

import torch as th

from eta_ctrl.util import dict_get_any

from .sb3_extensions import processors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from stable_baselines3.common.vec_env import VecEnv, VecNormalize

    from eta_ctrl.envs import BaseEnv
    from eta_ctrl.util.type_annotations import Path
from logging import getLogger

log = getLogger(__name__)


def deserialize_net_arch(
    net_arch: Sequence[Mapping[str, Any]], in_features: int, device: th.device | str = "auto"
) -> th.nn.Sequential:
    """Deserialize_net_arch can take a list of dictionaries describing a sequential torch network and deserialize
    it by instantiating the corresponding classes.

    An example for a possible net_arch would be:

    .. code-block::

        [{"layer": "Linear", "out_features": 60},
         {"activation_func": "Tanh"},
         {"layer": "Linear", "out_features": 60},
         {"activation_func": "Tanh"}]

    One key of the dictionary should be either 'layer', 'activation_func' or 'process'. If the 'layer' key is present,
    a layer from the :py:mod:`torch.nn` module is instantiated, if the 'activation_func' key is present, the
    value will be instantiated as an activation function from :py:mod:`torch.nn`. If the key 'process' is present,
    the value will be interpreted as a data processor from :py:mod:`eta_ctrl.common.processors`.

    All other keys of each dictionary will be used as keyword parameters to the instantiation of the layer,
    activation function or processor.

    Only the number of input features for the first layer must be specified (using the 'in_features') parameter.
    The function will then automatically determine the number of input features for all other layers in the
    sequential network.

    :param net_arch: List of dictionaries describing the network architecture.
    :param in_features: Number of input features for the first layer.
    :param device: Torch device to use for training the network.
    :return: Sequential torch network.
    """
    network = th.nn.Sequential()
    _features = in_features

    for net in net_arch:
        _net = dict(net)
        if "process" in net:
            process = getattr(processors, _net.pop("process"))

            # The "Split" process must be treated differently, because it needs to be deserialized recursively.
            if {"net_arch" and "sizes"} < inspect.signature(process).parameters.keys():
                sizes = process.get_full_sizes(_features, _net["sizes"])
                _net["net_arch"] = [deserialize_net_arch(e, sizes[i], device) for i, e in enumerate(_net["net_arch"])]

            try:
                if len({"in_channels", "in_features"} & inspect.signature(process).parameters.keys()) > 0:
                    network.append(process(_features, **_net))
                else:
                    network.append(process(**_net))
            except TypeError as e:
                msg = f"Could not instantiate processing module {process.__name__}: {e}"
                raise TypeError(msg) from e

        elif "layer" in net:
            layer = getattr(th.nn, _net.pop("layer"))

            # Set the number of input features if required by the layer class
            try:
                if len({"in_channels", "in_features"} & inspect.signature(layer).parameters.keys()) > 0:
                    network.append(layer(_features, **_net))
                else:
                    network.append(layer(**_net))
            except TypeError as e:
                msg = f"Could not instantiate layer module {layer.__name__}: {e}"
                raise TypeError(msg) from e

        elif "activation_func" in net:
            activation_func = _net.pop("activation_func")
            try:
                network.append(getattr(th.nn, activation_func)(**_net))
            except TypeError as e:
                msg = f"Could not instantiate activation function module {activation_func}: {e}"
                raise TypeError(msg) from e
        else:
            msg = f"Unknown process or layer type: {net}."
            raise ValueError(msg)

        _features = dict_get_any(_net, "out_channels", "out_features", fail=False, default=_features)

    network.to(device)
    return network


def is_vectorized(env: BaseEnv | VecEnv | VecNormalize | None) -> bool:
    """Check if an environment is vectorized.

    :param env: The environment to check.
    """
    if env is None:
        return False

    return hasattr(env, "num_envs")


def is_closed(env: BaseEnv | VecEnv | VecNormalize | None) -> bool:
    """Check whether an environment has been closed.

    :param env: The environment to check.
    """
    if env is None:
        return True

    if hasattr(env, "closed"):
        return env.closed

    if hasattr(env, "venv"):
        return is_closed(env.venv)

    return False


def episode_results_path(series_results_path: Path, run_name: str, episode: int, env_id: int = 1) -> pathlib.Path:
    """Generate a filepath which can be used for storing episode results of a specific environment as a csv file.

    Name is of the format: ThisRun_001_01.csv (run name _ episode number _ environment id .csv)

    :param series_results_path: Path for results of the series of optimization runs.
    :param run_name: Name of the optimization run.
    :param episode: Number of the episode the environment is working on.
    :param env_id: Identification of the environment.
    """
    path = series_results_path if isinstance(series_results_path, pathlib.Path) else pathlib.Path(series_results_path)

    return path / f"{episode_name_string(run_name, episode, env_id)}.csv"


def episode_name_string(run_name: str, episode: int, env_id: int = 1) -> str:
    """Generate a name which can be used to pre or postfix files from a specific episode and run of an environment.

    Name is of the format: ThisRun_001_01 (run name _ episode number _ environment id)

    :param run_name: Name of the optimization run.
    :param episode: Number of the episode the environment is working on.
    :param env_id: Identification of the environment.
    """
    return f"{run_name}_{episode:0>#3}_{env_id:0>#2}"
