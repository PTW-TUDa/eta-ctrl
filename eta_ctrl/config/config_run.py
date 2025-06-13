from __future__ import annotations

import pathlib
from logging import getLogger
from typing import TYPE_CHECKING

from attrs import converters, define, field, validators

if TYPE_CHECKING:
    from eta_ctrl.envs import BaseEnv


log = getLogger(__name__)


@define(frozen=True, kw_only=True)
class ConfigRun:
    """Configuration for an optimization run, including the series and run names descriptions and paths
    for the run.
    """

    #: Name of the series of optimization runs.
    series: str = field(validator=validators.instance_of(str))
    #: Name of an optimization run.
    name: str = field(validator=validators.instance_of(str))
    #: Description of an optimization run.
    description: str = field(
        converter=converters.default_if_none(""),  # type: ignore[misc]
        validator=validators.instance_of(str),
    )
    #: Root path of the framework run.
    path_root: pathlib.Path = field(converter=pathlib.Path)
    #: Path to results of the optimization run.
    path_results: pathlib.Path = field(converter=pathlib.Path)
    #: Path to scenarios used for the optimization run.
    path_scenarios: pathlib.Path | None = field(default=None, converter=converters.optional(pathlib.Path))
    #: Path for the results of the series of optimization runs.
    path_series_results: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the model of the optimization run.
    path_run_model: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to information about the optimization run.
    path_run_info: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the monitoring information about the optimization run.
    path_run_monitor: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the normalization wrapper information.
    path_vec_normalize: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the neural network architecture file.
    path_net_arch: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the log output file.
    path_log_output: pathlib.Path = field(init=False, converter=pathlib.Path)

    # Information about the environments
    #: Version of the main environment.
    env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the main environment.
    env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    #: Version of the secondary environment (interaction_env).
    interaction_env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the secondary environment (interaction_env).
    interaction_env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    def __attrs_post_init__(self) -> None:
        """Add default values to the derived paths."""
        object.__setattr__(self, "path_series_results", self.path_results / self.series)
        object.__setattr__(self, "path_run_model", self.path_series_results / f"{self.name}_model.zip")
        object.__setattr__(self, "path_run_info", self.path_series_results / f"{self.name}_info.json")
        object.__setattr__(self, "path_run_monitor", self.path_series_results / f"{self.name}_monitor.csv")
        object.__setattr__(self, "path_vec_normalize", self.path_series_results / "vec_normalize.pkl")
        object.__setattr__(self, "path_net_arch", self.path_series_results / "net_arch.txt")
        object.__setattr__(self, "path_log_output", self.path_series_results / f"{self.name}_log_output.log")

    def create_results_folders(self) -> None:
        """Create the results folders for an optimization run (or check if they already exist)."""
        if not self.path_results.is_dir():
            for p in reversed(self.path_results.parents):
                if not p.is_dir():
                    p.mkdir()
                    log.info(f"Directory created: \n\t {p}")
            self.path_results.mkdir()
            log.info(f"Directory created: \n\t {self.path_results}")

        if not self.path_series_results.is_dir():
            log.debug("Path for result series doesn't exist on your OS. Trying to create directories.")
            self.path_series_results.mkdir()
            log.info(f"Directory created: \n\t {self.path_series_results}")

    def set_env_info(self, env: type[BaseEnv]) -> None:
        """Set the environment information of the optimization run to represent the given environment.
        The information will default to None if this is never called.

        :param env: The environment whose description should be used.
        """
        version, description = env.get_info()
        object.__setattr__(self, "env_version", version)
        object.__setattr__(self, "env_description", description)

    def set_interaction_env_info(self, env: type[BaseEnv]) -> None:
        """Set the interaction environment information of the optimization run to represent the given environment.
        The information will default to None if this is never called.

        :param env: The environment whose description should be used.
        """
        version, description = env.get_info()
        object.__setattr__(self, "interaction_env_version", version)
        object.__setattr__(self, "interaction_env_description", description)

    @property
    def paths(self) -> dict[str, pathlib.Path]:
        """Dictionary of all paths for the optimization run. This is for easier access and contains all
        paths as mentioned above."""
        paths = {
            "path_root": self.path_root,
            "path_results": self.path_results,
            "path_series_results": self.path_series_results,
            "path_run_model": self.path_run_model,
            "path_run_info": self.path_run_info,
            "path_run_monitor": self.path_run_monitor,
            "path_vec_normalize": self.path_vec_normalize,
            "path_log_output": self.path_log_output,
        }
        if self.path_scenarios is not None:
            paths["path_scenarios"] = self.path_scenarios

        return paths
