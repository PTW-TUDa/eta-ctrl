from __future__ import annotations

import pathlib
from csv import DictWriter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict

from eta_ctrl.util.io_utils import load_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from typing_extensions import Self

    from eta_ctrl.util.type_annotations import Path
from logging import getLogger

log = getLogger(__name__)


class StateVar(BaseModel):
    """A variable in the state of an environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Name of the state variable (This must always be specified).
    name: str

    #: Should the agent specify actions for this variable? (default: False).
    is_agent_action: bool = False
    #: Should the agent be allowed to observe the value of this variable? (default: False).
    is_agent_observation: bool = False
    #: Should the state log of this episode be added to state_log_longtime? (default: True).
    add_to_state_log: bool = True

    #: Name of the variable in the external interaction model
    #: (e.g.: environment or FMU) (default: StateVar.name if (is_ext_input or is_ext_output) else None).
    ext_id: str | None = None
    #: Should this variable be passed to the external model as an input? (default: False).
    is_ext_input: bool = False
    #: Should this variable be parsed from the external model output? (default: False).
    is_ext_output: bool = False
    #: Value to add to the output from an external model (default: 0.0).
    ext_scale_add: float = 0.0
    #: Value to multiply to the output from an external model (default: 1.0).
    ext_scale_mult: float = 1.0

    #: Name or identifier (order) of the variable in an interaction environment (default: None).
    interact_id: int | None = None
    #: Should this variable be read from the interaction environment? (default: False).
    from_interact: bool = False
    #: Value to add to the value read from an interaction (default: 0.0).
    interact_scale_add: float = 0.0
    #: Value to multiply to the value read from  an interaction (default: 1.0).
    interact_scale_mult: float = 1.0

    #: Name of the scenario variable, this value should be read from (default: None).
    scenario_id: str | None = None
    #: Should this variable be read from imported timeseries date? (default: False).
    from_scenario: bool = False
    #: Value to add to the value read from a scenario file (default: 0.0).
    scenario_scale_add: float = 0.0
    #: Value to multiply to the value read from a scenario file (default: 1.0).
    scenario_scale_mult: float = 1.0

    #: Lowest possible value of the state variable (default: -np.inf).
    low_value: float = -np.inf
    #: Highest possible value of the state variable (default: np.inf).
    high_value: float = np.inf
    #: If the value of the variable dips below this, the episode should be aborted (default: -np.inf).
    abort_condition_min: float = -np.inf
    #: If the value of the variable rises above this, the episode should be aborted (default: np.inf).
    abort_condition_max: float = np.inf

    #: Determine the index, where to look (useful for mathematical optimization, where multiple time steps could be
    #: returned). In this case, the index values might be different for actions and observations.
    index: int = 0

    def model_post_init(self, context: Any) -> None:
        if (self.is_ext_input or self.is_ext_output) and self.ext_id is None:
            object.__setattr__(self, "ext_id", self.name)
            log.info(f"Using name as ext_id for variable {self.name}")

        if (not self.from_interact) ^ (self.interact_id is None):
            msg = f"Variable {self.name} is either missing `interact_id` or `from_interact`."
            raise KeyError(msg)
        if (not self.from_scenario) ^ (self.scenario_id is None):
            msg = f"Variable {self.name} is either missing `scenario_id` or `from_scenario`."
            raise KeyError(msg)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | pd.Series) -> StateVar:
        """Initialize a state var from a dictionary or pandas Series.

        :param mapping: dictionary or pandas Series to initialize from.
        :return: Initialized StateVar object
        """
        return cls(**dict(mapping))

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __str__(self) -> str:
        """Human-readable string representation of StateVar."""
        var_type = []
        if self.is_agent_action:
            var_type.append("action")
        if self.is_agent_observation:
            var_type.append("observation")
        if not var_type:
            var_type.append("variable")

        type_str = "/".join(var_type)
        has_range = self.low_value != -np.inf or self.high_value != np.inf
        range_str = f"[{self.low_value}, {self.high_value}]" if has_range else ""

        return f"StateVar '{self.name}' ({type_str}){' ' + range_str if range_str else ''}"

    def __repr__(self) -> str:
        """Developer-friendly string representation of StateVar."""
        key_attrs = []
        if self.is_agent_action:
            key_attrs.append("is_agent_action=True")
        if self.is_agent_observation:
            key_attrs.append("is_agent_observation=True")
        if self.low_value != -np.inf:
            key_attrs.append(f"low_value={self.low_value}")
        if self.high_value != np.inf:
            key_attrs.append(f"high_value={self.high_value}")

        attrs_str = ", ".join(key_attrs)
        return f"StateVar(name='{self.name}'{', ' + attrs_str if attrs_str else ''})"


class StateStructure(BaseModel):
    """Used for parsing the state structure from a config file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_parameters: dict[str, float | bool] | None = None
    actions: list[StateVar]
    observations: list[StateVar]


class StateConfig:
    """The configuration for the action and observation spaces. The values are used to control which variables are
    part of the action space and observation space. Additionally, the parameters can specify abort conditions
    and the handling of values from interaction environments or from simulation. Therefore, the *StateConfig*
    is very important for the functionality of EtaCtrl.
    """

    def __init__(self, *state_vars: StateVar) -> None:
        #: Mapping of the variables names to their StateVar instance with all associated information.
        self.vars = {var.name: var for var in state_vars}
        # Additional Dataframe for easier access
        if state_vars:
            self.df_vars: pd.DataFrame = pd.DataFrame([var.model_dump() for var in state_vars]).set_index("name")
            if not self.df_vars.index.is_unique:
                duplicates = self.df_vars.index[self.df_vars.index.duplicated()].unique().tolist()
                msg = f"Duplicate variable names in StateConfig: {duplicates}"
                raise ValueError(msg)
        else:
            # Handle empty case - create empty DataFrame with expected columns
            self.df_vars = pd.DataFrame(columns=list(StateVar.model_fields.keys())).set_index("name")

        #: List of variables that are agent actions. Needs to be ordered.
        self.actions: list[str] = self.df_vars.query("is_agent_action == True").index.tolist()
        #: Set of variables that are agent observations.
        self.observations: list[str] = self.df_vars.query("is_agent_observation == True").index.tolist()
        #: Set of variables that should be logged.
        self.add_to_state_log: list[str] = self.df_vars.query("add_to_state_log == True").index.tolist()

        #: List of variables that should be provided to an external source (such as an FMU).
        self.ext_inputs: list[str] = self.df_vars.query("is_ext_input == True").index.tolist()
        #: List of variables that can be received from an external source (such as an FMU).
        self.ext_outputs: list[str] = self.df_vars.query("is_ext_output == True").index.tolist()
        #: Mapping of variable names to their external IDs.
        self.map_ext_ids: dict[str, str] = self.df_vars.query("ext_id != None").ext_id.to_dict()
        #: Reverse mapping of external IDs to their corresponding variable names.
        self.rev_ext_ids: dict[str, str] = {v: k for k, v in self.map_ext_ids.items()}

        #: List of variables that should be read from an interaction environment.
        self.interact_outputs: list[str] = self.df_vars.query("from_interact == True").index.tolist()
        #: Mapping of internal environment names to interact IDs.
        self.map_interact_ids: dict[str, str] = self.df_vars["interact_id"].to_dict()

        #: List of variables which are loaded from scenario files.
        self.scenarios: list[str] = self.df_vars.query("from_scenario == True").index.tolist()
        #: Mapping of internal environment names to scenario IDs.
        self.map_scenario_ids: dict[str, str] = self.df_vars["scenario_id"].to_dict()

        #: List of variables that have minimum values for an abort condition.
        self.abort_conditions_min: list[str] = self.df_vars["abort_condition_min"].dropna().index.tolist()
        #: List of variables that have maximum values for an abort condition.
        self.abort_conditions_max: list[str] = self.df_vars["abort_condition_max"].index.tolist()

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a StateConfig from a config file.

        :param path: Path of the config file.
        :return: StateConfig object.
        """
        raw_dict = load_config(path)

        actions: list[dict[str, Any]] = raw_dict.get("actions") or []
        observations: list[dict[str, Any]] = raw_dict.get("observations") or []

        actions = [{**act, "is_agent_action": True} for act in actions]
        observations = [{**obs, "is_agent_observation": True} for obs in observations]

        all_states = actions + observations

        if len(all_states) == 0:
            msg = f"Invalid StateConfig at {path} with no StateVar's"
            raise ValueError(msg)

        # Defined by user in *structure.toml
        state_params = raw_dict.get("state_parameters")
        if isinstance(state_params, dict):
            log.debug(f"Using State parameters {state_params} from {path} for StateConfig.")
        else:
            log.warning(f"State parameters in {path} need to be a dict!")
            return cls.from_dict(mapping=all_states)

        return cls.from_dict(mapping=all_states, state_params=state_params)

    @classmethod
    def from_dict(
        cls, mapping: Sequence[dict[str, Any]] | pd.DataFrame, *, state_params: dict[str, float] | None = None
    ) -> Self:
        """Convert a potentially incomplete StateConfig DataFrame or a list of dictionaries to the
        standardized StateConfig format. This will ignore any additional columns.

        :param mapping: Mapping to be converted to the StateConfig format.
        :param state_params: State parameter values for parameters supplied in mapping (e.g. {min_temp: 20})
        :return: StateConfig object.
        """
        _mapping = mapping.to_dict("records") if isinstance(mapping, pd.DataFrame) else mapping
        if not state_params:
            state_params = {}

        for statevar in _mapping:
            for field_name, value in statevar.items():
                if field_name in ("name", "ext_id", "scenario_id"):  # Supposed to be strings
                    continue
                if isinstance(value, str):
                    parameter_name = value
                    if is_negative := value.startswith("-"):
                        parameter_name = parameter_name[1:]  # strip minus sign

                    new_value = state_params.get(parameter_name)
                    if new_value is None:
                        msg = f"Parameter {parameter_name} needs to be specified in state_params."
                        raise KeyError(msg)
                    if is_negative:
                        new_value = -new_value
                    statevar[field_name] = new_value

        return cls(*[StateVar.from_dict(col) for col in _mapping])

    def store_file(self, file: Path) -> None:
        """Save the StateConfig to a comma separated file.

        :param file: Path to the file.
        """
        _file = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        _header = StateVar.model_fields.keys()

        with _file.open("w") as f:
            writer = DictWriter(f, _header, restval="None", delimiter=";")
            writer.writeheader()
            for var in self.vars.values():
                writer.writerow(var.model_dump())

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:
        """Check whether the given state is within the abort conditions specified by the StateConfig instance.

        :param state: The state array to check for conformance.
        :return: Result of the check (False if the state does not conform to the required conditions).
        """
        valid_min = all(state[name] >= self.vars[name].abort_condition_min for name in state)
        if not valid_min:
            log.warning("Minimum abort condition exceeded by at least one value.")

        valid_max = all(state[name] <= self.vars[name].abort_condition_max for name in state)
        if not valid_max:
            log.warning("Maximum abort condition exceeded by at least one value.")

        return valid_min and valid_max

    def continuous_action_space(self) -> spaces.Box:
        """Generate a numpy ndarray action space.

        :return: Action space.
        """
        actions = self.df_vars.query("is_agent_action == True")
        low_values = actions["low_value"].to_numpy(dtype=np.float32)
        high_values = actions["high_value"].to_numpy(dtype=np.float32)

        return spaces.Box(low_values, high_values)

    def continuous_observation_space(self) -> spaces.Dict:
        """Generate a dictionary observation space.

        :return: Observation Space.
        """
        observations: dict[str, spaces.Box] = {
            name: spaces.Box(low=row["low_value"], high=row["high_value"], dtype=np.float32)
            for name, row in self.df_vars.iterrows()
            if row["is_agent_observation"] is True
        }
        return spaces.Dict(observations)  # type: ignore[arg-type]

    def continuous_spaces(self) -> tuple[spaces.Box, spaces.Dict]:
        """Generate continuous action and observation spaces according to the OpenAI specification.

        :return: Tuple of action space and observation space.
        """
        action_space = self.continuous_action_space()
        observation_space = self.continuous_observation_space()
        return action_space, observation_space

    def __str__(self) -> str:
        """Human-readable string representation of StateConfig."""
        n_actions = len(self.actions)
        n_observations = len(self.observations)
        n_total = len(self.vars)

        return f"StateConfig with {n_actions} actions, {n_observations} observations ({n_total} total variables)"

    def __repr__(self) -> str:
        """Developer-friendly string representation of StateConfig."""
        n_actions = len(self.actions)
        n_observations = len(self.observations)

        # Show first few variables for context
        actions_preview = self.actions[:3]
        observations_preview = list(self.observations)[:3]

        actions_str = f"{actions_preview}{'...' if n_actions > 3 else ''}"
        observations_str = f"{observations_preview}{'...' if n_observations > 3 else ''}"

        return f"StateConfig(actions={actions_str}, observations={observations_str})"

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @property
    def loc(self) -> pd.api.indexers._LocIndexer:
        """Behave like dataframe (enable indexing via loc) for compatibility."""
        return self.vars
