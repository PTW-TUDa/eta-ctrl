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
    root_path: pathlib.Path = field(converter=pathlib.Path)
    #: Path to results of the optimization run.
    results_path: pathlib.Path = field(converter=pathlib.Path)
    #: Path to scenarios used for the optimization run.
    scenarios_path: pathlib.Path = field(converter=pathlib.Path)
    #: Path for the results of the series of optimization runs.
    series_results_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the model of the optimization run.
    run_model_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to information about the optimization run.
    run_info_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the monitoring information about the optimization run.
    run_monitor_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the normalization wrapper information.
    vec_normalize_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the neural network architecture file.
    net_arch_path: pathlib.Path = field(init=False, converter=pathlib.Path)
    #: Path to the log output file.
    log_output_path: pathlib.Path = field(init=False, converter=pathlib.Path)

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
        object.__setattr__(self, "series_results_path", self.results_path / self.series)
        object.__setattr__(self, "run_model_path", self.series_results_path / f"{self.name}_model.zip")
        object.__setattr__(self, "run_info_path", self.series_results_path / f"{self.name}_info.json")
        object.__setattr__(self, "run_monitor_path", self.series_results_path / f"{self.name}_monitor.csv")
        object.__setattr__(self, "vec_normalize_path", self.series_results_path / "vec_normalize.pkl")
        object.__setattr__(self, "net_arch_path", self.series_results_path / "net_arch.txt")
        object.__setattr__(self, "log_output_path", self.series_results_path / f"{self.name}_log_output.log")

    def create_results_folders(self) -> None:
        """Create the results folders for an optimization run (or check if they already exist)."""
        if not self.results_path.is_dir():
            for p in reversed(self.results_path.parents):
                if not p.is_dir():
                    p.mkdir()
                    log.info(f"Directory created: \n\t {p}")
            self.results_path.mkdir()
            log.info(f"Directory created: \n\t {self.results_path}")

        if not self.series_results_path.is_dir():
            log.debug("Path for result series doesn't exist on your OS. Trying to create directories.")
            self.series_results_path.mkdir()
            log.info(f"Directory created: \n\t {self.series_results_path}")

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
            "root_path": self.root_path,
            "results_path": self.results_path,
            "series_results_path": self.series_results_path,
            "run_model_path": self.run_model_path,
            "run_info_path": self.run_info_path,
            "run_monitor_path": self.run_monitor_path,
            "vec_normalize_path": self.vec_normalize_path,
            "log_output_path": self.log_output_path,
        }
        if self.scenarios_path is not None:
            paths["scenarios_path"] = self.scenarios_path

        return paths
