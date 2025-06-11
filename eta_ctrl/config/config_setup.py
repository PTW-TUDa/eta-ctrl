from __future__ import annotations

import importlib
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import converters, define, field, fields

if TYPE_CHECKING:
    from typing import Any

    from attrs import Attribute
    from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
    from stable_baselines3.common.vec_env import DummyVecEnv

    from eta_ctrl.envs import BaseEnv


log = getLogger(__name__)


def _get_class(instance: ConfigSetup, attrib: Attribute, new_value: str | None) -> str | None:
    """Find module and class name and import the specified class."""
    if new_value is not None:
        module, cls_name = new_value.rsplit(".", 1)
        try:
            cls = getattr(importlib.import_module(module), cls_name)
        except ModuleNotFoundError as e:
            msg = f"Could not find module '{e.name}'. While importing class '{cls_name}' from '{attrib.name}' value."
            raise ModuleNotFoundError(msg) from e
        except AttributeError as e:
            msg = (
                f"Could not find class '{cls_name}' in module '{module}'. "
                f"While importing class '{cls_name}' from '{attrib.name}' value."
            )
            raise AttributeError(msg) from e

        cls_attr_name = f"{attrib.name.rsplit('_', 1)[0]}_class"
        setattr(instance, cls_attr_name, cls)

    return new_value


@define(frozen=False, kw_only=True)
class ConfigSetup:
    """Configuration options as specified in the "setup" section of the configuration file."""

    #: Import description string for the agent class.
    agent_import: str = field(on_setattr=_get_class)
    #: Agent class (automatically determined from agent_import).
    agent_class: type[BaseAlgorithm] = field(init=False)
    #: Import description string for the environment class.
    environment_import: str = field(on_setattr=_get_class)
    #: Imported Environment class (automatically determined from environment_import).
    environment_class: type[BaseEnv] = field(init=False)
    #: Import description string for the interaction environment (default: None).
    interaction_env_import: str | None = field(default=None, on_setattr=_get_class)
    #: Interaction environment class (default: None) (automatically determined from interaction_env_import).
    interaction_env_class: type[BaseEnv] | None = field(init=False, default=None)

    #: Import description string for the environment vectorizer
    #: (default: stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv).
    vectorizer_import: str = field(
        default="stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv",
        on_setattr=_get_class,
        converter=converters.default_if_none(  # type: ignore[misc]
            "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"
        ),
    )  # mypy currently does not recognize converters.default_if_none
    #: Environment vectorizer class  (automatically determined from vectorizer_import).
    vectorizer_class: type[DummyVecEnv] = field(init=False)
    #: Import description string for the policy class (default: eta_ctrl.agents.common.NoPolicy).
    policy_import: str = field(
        default="eta_ctrl.common.NoPolicy",
        on_setattr=_get_class,
        converter=converters.default_if_none("eta_ctrl.common.NoPolicy"),  # type: ignore[misc]
    )  # mypy currently does not recognize converters.default_if_none
    #: Policy class (automatically determined from policy_import).
    policy_class: type[BasePolicy] = field(init=False)

    #: Flag which is true if the environment should be wrapped for monitoring (default: False).
    monitor_wrapper: bool = field(default=False, converter=bool)
    #: Flag which is true if the observations should be normalized (default: False).
    norm_wrapper_obs: bool = field(default=False, converter=bool)
    #: Flag which is true if the rewards should be normalized (default: False).
    norm_wrapper_reward: bool = field(default=False, converter=bool)
    #: Flag to enable tensorboard logging (default: False).
    tensorboard_log: bool = field(default=False, converter=bool)

    def __attrs_post_init__(self) -> None:
        _fields = fields(ConfigSetup)
        _get_class(self, _fields.agent_import, self.agent_import)
        _get_class(self, _fields.environment_import, self.environment_import)
        _get_class(self, _fields.interaction_env_import, self.interaction_env_import)
        _get_class(self, _fields.vectorizer_import, self.vectorizer_import)
        _get_class(self, _fields.policy_import, self.policy_import)

    @classmethod
    def from_dict(cls, dikt: dict[str, Any]) -> ConfigSetup:
        errors = []

        def get_import(name: str, required: bool = False) -> str | Any:
            """Get import string or combination of package and class name from dictionary.
            :param name: Name of the configuration value.
            :param required: Flag to determine if the value is required.
            """
            nonlocal errors, dikt
            import_value = dikt.pop(f"{name}_import", None)
            package_value = dikt.pop(f"{name}_package", None)
            class_value = dikt.pop(f"{name}_class", None)
            # Check import
            if import_value is not None:
                return import_value

            # Check package and class
            if package_value is not None and class_value is not None:
                return f"{package_value}.{class_value}"

            # If only one of package and class is specified, raise error
            if (package_value is None) ^ (class_value is None):
                msg = f"Only one of '{name}_package' and '{name}_class' is specified."
                log.info(msg)

            # Raise error if required value is missing
            if required:
                msg = f"'{name}_import' or both of '{name}_package' and '{name}_class' parameters must be specified."
                log.error(msg)
                errors.append(name)
            return None

        agent_import = get_import("agent", required=True)
        environment_import = get_import("environment", required=True)

        interaction_env_import = get_import("interaction_env")
        vectorizer_import = get_import("vectorizer")
        policy_import = get_import("policy")

        monitor_wrapper = dikt.pop("monitor_wrapper", None)
        norm_wrapper_obs = dikt.pop("norm_wrapper_obs", None)
        norm_wrapper_reward = dikt.pop("norm_wrapper_reward", None)
        tensorboard_log = dikt.pop("tensorboard_log", None)

        # Log configuration values which were not recognized.
        if dikt:
            msg = "Following values were not recognized in the config setup section and are ignored: "
            msg += ", ".join(dikt.keys())
            log.warning(msg)

        if errors:
            msg = "Not all required values were found in setup section (see log). Could not load config file. "
            msg += f"Missing values: {', '.join(errors)}"
            raise ValueError(msg)

        return ConfigSetup(
            agent_import=agent_import,
            environment_import=environment_import,
            interaction_env_import=interaction_env_import,
            vectorizer_import=vectorizer_import,
            policy_import=policy_import,
            monitor_wrapper=monitor_wrapper,
            norm_wrapper_obs=norm_wrapper_obs,
            norm_wrapper_reward=norm_wrapper_reward,
            tensorboard_log=tensorboard_log,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            msg = f"The key {name} does not exist - it cannot be set."
            raise KeyError(msg)
        setattr(self, name, value)
