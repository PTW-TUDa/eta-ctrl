from __future__ import annotations

import importlib
from logging import getLogger
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, model_validator
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from eta_ctrl.envs import BaseEnv

if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue


log = getLogger(__name__)


def _import_class(import_path: str, expected_base: type) -> type:
    """Import a class from *import_path* and verify it is a subclass of *expected_base*."""
    module, cls_name = import_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)
    if not issubclass(cls, expected_base):
        msg = f"'{import_path}' resolved to {cls}, which is not a subclass of {expected_base.__name__}"
        raise TypeError(msg)
    return cls


class ConfigSetup(BaseModel):
    """Helper class, which is part of `Config`, for import and setup parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    agent_import: str
    """Import description string for the agent class."""
    agent_class: type[BaseAlgorithm] = Field(exclude=True)
    """Agent class (automatically determined from agent_import)."""
    environment_import: str
    """Import description string for the environment class."""
    environment_class: type[BaseEnv] = Field(exclude=True)
    """Imported Environment class (automatically determined from environment_import)."""

    vectorizer_import: str = "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"
    """Import description string for the environment vectorizer
    (default: stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv)."""
    vectorizer_class: type[DummyVecEnv | SubprocVecEnv] = Field(exclude=True)
    """Environment vectorizer class (automatically determined from vectorizer_import)."""

    policy_import: str = "eta_ctrl.common.NoPolicy"
    """Import description string for the policy class (default: eta_ctrl.agents.common.NoPolicy)."""
    policy_class: type[BasePolicy] = Field(exclude=True)
    """Policy class (automatically determined from policy_import)."""

    monitor_wrapper: bool = False
    """Flag which is true if the environment should be wrapped for monitoring (default: False)."""
    norm_wrapper_obs: bool = False
    """Flag which is true if the observations should be normalized (default: False)."""
    norm_wrapper_reward: bool = False
    """Flag which is true if the rewards should be normalized (default: False)."""
    tensorboard_log: bool = False
    """Flag to enable tensorboard logging (default: False)."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_classes(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["agent_class"] = _import_class(data["agent_import"], BaseAlgorithm)
        data["environment_class"] = _import_class(data["environment_import"], BaseEnv)
        data["vectorizer_class"] = _import_class(
            data.get("vectorizer_import", "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"),
            VecEnv,
        )
        data["policy_class"] = _import_class(
            data.get("policy_import", "eta_ctrl.common.NoPolicy"),
            BasePolicy,
        )
        return data

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        # Remove resolved class fields — these are set automatically from the _import fields
        for field in ("agent_class", "environment_class", "vectorizer_class", "policy_class"):
            json_schema.get("properties", {}).pop(field, None)
            if "required" in json_schema:
                json_schema["required"] = [r for r in json_schema["required"] if r != field]
        return json_schema

    def __str__(self) -> str:
        """Human-readable string representation of ConfigSetup."""
        return f"ConfigSetup(env={self.environment_class.__name__}, agent={self.agent_class.__name__})"
