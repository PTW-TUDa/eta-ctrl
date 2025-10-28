"""The FMUSimulator class enables easy simulation of FMU files."""

from __future__ import annotations

import pathlib
import shutil
from contextlib import contextmanager
from logging import getLogger
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Model, FMU2Slave
from fmpy.sundials import CVodeSolver
from fmpy.util import compile_platform_binary

from eta_ctrl.util.utils import timestep_to_seconds

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Any

    from fmpy.model_description import ModelDescription

    from eta_ctrl.util.type_annotations import Number, Path, TimeStep

log = getLogger(__name__)


class FMUSimulator:
    """FMU simulator object.

    :param _id: FMU instance ID.
    :param fmu_path: Path to the FMU file.
    :param start_time: Simulation start time in seconds.
    :param float stop_time: Simulation stop time in seconds.
    :param step_size: Simulation step size in seconds.
    :param names_inputs: List of input names that correspond to names used in the FMU file (e.g. ['u', 'p']).
                         If the step function is going to be used with lists as input values, this list will be used
                         to translate between the list position and the variable name in the FMU.
    :param names_outputs: List of output names that correspond to names used in the FMU file
                          (e.g. ['y', 'th', 'thdot']). If the step function should return only specific values instead
                          of all results as a dictionary, this parameter can be specified to determine, which parameters
                          should be returned.
    :param init_values: Starting values for parameters that should be pushed to the FMU with names corresponding to
                        variables in the FMU.
    """

    def __init__(
        self,
        _id: int,
        fmu_path: Path,
        start_time: TimeStep = 0,
        stop_time: TimeStep = 1,
        step_size: TimeStep = 1,
        names_inputs: Sequence[str] | None = None,
        names_outputs: Sequence[str] | None = None,
        init_values: Mapping[str, float] | None = None,
    ) -> None:
        #: Path to the FMU model.
        self.fmu_path = fmu_path

        #: Start time for the simulation in time increments.
        self.start_time = int(timestep_to_seconds(start_time))
        #: Stopping time for the simulation in time increments (only relevant if run in simulation loop).
        self.stop_time = int(timestep_to_seconds(stop_time))
        #: Step size (time) for the simulation in time increments.
        self.step_size = int(timestep_to_seconds(step_size))

        #: Model description from the FMU (contains variable names, types, references and more).
        self.model_description: ModelDescription = read_model_description(fmu_path)

        #: Variable map from model description. The map specifies the value reference and datatype of a named
        #: variable in the FMU. The structure is {'name': {'ref': <value reference>, 'type': <variable data type>}}.
        self.__type_map = {"Real": "real", "Boolean": "bool", "Integer": "int", "Enumeration": "enum"}

        self._model_vars: dict[str, dict[str, str]] = {}
        for var in self.model_description.modelVariables:
            self._model_vars[var.name] = {"ref": var.valueReference, "type": self.__type_map[var.type]}
        self.model_vars_names = list(self._model_vars.keys())

        if names_inputs is not None:
            self._validate_names(names_inputs, context="input names")
            input_names = names_inputs
        else:
            input_names = self.model_vars_names
        #: Mapping of input names to model references
        self.input_mapping = {name: self._model_vars[name]["ref"] for name in input_names}

        if names_outputs is not None:
            self._validate_names(names_outputs, context="output names")
            output_names = names_outputs
        else:
            output_names = self.model_vars_names
        #: Mapping of output names to model references
        self.output_mapping = {name: self._model_vars[name]["ref"] for name in output_names}

        #: Directory where the FMU will be extracted.
        self._unzipdir: Path = extract(fmu_path)

        try:
            #: Instance of the FMU Slave object.
            self.fmu: FMU2Slave = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self._unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName="FMUsimulator_" + str(_id),
            )
        except Exception:  # noqa: BLE001  fmpy raises bare Exceptions
            compile_platform_binary(self.fmu_path)
            self.fmu = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self._unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName="FMUsimulator_" + str(_id),
            )

        # initialize
        self.fmu.instantiate(visible=False, callbacks=None, loggingOn=False)
        self.fmu.setupExperiment(startTime=self.start_time)

        # set init values
        # instead of using the fmpy apply_start_values func from fmpy use the own set_values func to set the values
        # of the simulation variables correctly, reasons are also performance and simulation speed
        init_values = {} if init_values is None else init_values
        self.set_values(init_values)

        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        #: Current simulation time.
        self.time = self.start_time

    def _validate_names(self, names: Sequence[str], context: str = "names") -> None:
        """Validate a given sequence of names by checking their existence in the FMU.

        :param names: Names to check.
        :param context: Additional context for log messages, defaults to "names".
        :raises KeyError: Unknown variable names.
        """
        names_set = set(names)
        model_vars_set = set(self.model_vars_names)
        if len(names_set) < len(names):
            log.warning(f"{context.capitalize()} contain duplicate elements")

        # Find names that don't exist in the model
        missing_names = names_set - model_vars_set
        if missing_names:
            msg = f"Found {len(missing_names)} unknown variable names in {context}: {sorted(missing_names)}. "
            raise KeyError(msg)

    @property
    def input_vars(self) -> list[str]:
        """Ordered list of all available input variable names in the FMU."""
        return list(self.input_mapping.keys())

    @property
    def output_vars(self) -> list[str]:
        """Ordered list of all available output variable names in the FMU."""
        return list(self.output_mapping.keys())

    @property
    def parameter_vars(self) -> list[str]:
        """Get names of all available parameters in the FMU.

        :return: List of parameter variable names.
        """
        if not hasattr(self, "_parameter_vars"):
            # Extract parameter variables from model description
            self._parameter_vars = []
            for var in self.model_description.modelVariables:
                # Check if the variable is a parameter (or has causality="parameter")
                if hasattr(var, "causality") and var.causality == "parameter":
                    self._parameter_vars.append(var.name)
        return self._parameter_vars

    def read_values(self, names: Sequence[str] | None = None) -> dict[str, float]:
        """Return current values of the simulation without advancing a simulation step or the simulation time.

        :param names: Sequence of values to read from the FMU. If this is None (default), all available values will be
                      read.
        :return: Read values from the FMU mapped with their names.
        """
        # Find value references and names for the variables that should be read from the FMU
        if names is None:
            var_refs = self.output_mapping
        else:
            self._validate_names(names, context="output names")
            var_refs = {name: self.output_mapping[name] for name in names}

        # Get values from the FMU and convert to a dictionary
        output_values = self.fmu.getReal(list(var_refs.values()))
        return dict(zip(var_refs.keys(), output_values, strict=False))

    def set_values(self, values: Mapping[str, Number | bool]) -> None:
        """Set values of simulation variables without advancing a simulation step or the simulation time.

        :param values: Values that should be pushed to the FMU. Names of the input_values must correspond
                       to variables in the FMU.
        """
        self._validate_names(list(values.keys()), context="input names")
        vals: dict[str, list[Number | bool]] = {"real": [], "int": [], "bool": []}
        refs: dict[str, list[str]] = {"real": [], "int": [], "bool": []}
        for var, val in values.items():
            model_var = self._model_vars[var]
            refs[model_var["type"]].append(model_var["ref"])
            vals[model_var["type"]].append(val)

        if len(refs["real"]) > 0:
            self.fmu.setReal(refs["real"], vals["real"])
        if len(refs["int"]) > 0:
            self.fmu.setInteger(refs["int"], vals["int"])
        if len(refs["bool"]) > 0:
            self.fmu.setBoolean(refs["bool"], vals["bool"])

    def step(
        self,
        input_values: Mapping[str, float] | None = None,
        output_names: Sequence[str] | None = None,
        advance_time: bool = True,
        nr_substeps: int | None = None,
    ) -> dict[str, float]:
        """Simulate next time step in the FMU with defined input values and output values.

        :param input_values: Current values that should be pushed to the FMU. Names of the input_values must correspond
                             to variables in the FMU.
        :param advance_time: Decide if the FMUsimulator should add one timestep to the simulation time or not.
                                  This can be deactivated, if you just want to look at the result of a simulation step
                                  beforehand, without actually advancing simulation time.
        :param nr_substeps: if simulation steps are divided into substeps, this value will let the simulator know
                                that no time violation warning is necessary.
        :return: Resulting input and output values from the FMU with the keys named corresponding to the variables
                 in the FMU.
        """
        if input_values is not None:
            self.set_values(input_values)

        # put out warning for time limit violation, if self.time + self.step_size > self.stop_time + full step size
        if self.time + self.step_size > self.stop_time + (int(nr_substeps) if nr_substeps else 1) * self.step_size:
            log.warning(
                f"Simulation time {self.time + self.step_size} s exceeds specified stop time of "
                f"{self.stop_time} s. Proceed with care, simulation may become inaccurate."
            )

        # push input values to the FMU and do one timestep, doStep performs a step of certain size
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)

        # advance time
        if advance_time:
            self.time += self.step_size  # advance the time

        return self.read_values(output_names)

    @classmethod
    def simulate(
        cls,
        fmu_path: Path,
        start_time: TimeStep = 0,
        stop_time: TimeStep = 1,
        step_size: TimeStep = 1,
        init_values: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Instantiate a simulator with the specified FMU, perform simulation and return results.

        :param fmu_path: Path to the FMU file.
        :param start_time: Simulation start time in seconds.
        :param float stop_time: Simulation stop time in seconds.
        :param step_size: simulation step size in seconds.
        :param init_values: Starting values for parameters that should be pushed to the FMU with names corresponding to
                            variables in the FMU.
        """
        simulator = cls(0, fmu_path, start_time, stop_time, step_size, init_values=init_values)
        dt = np.dtype([(name, float) for name in simulator.read_values()])
        # mypy does not recognize the return type of floor division...
        result = np.rec.array(
            None,
            shape=((simulator.stop_time - simulator.start_time) // simulator.step_size + 1,),
            dtype=dt,
        )
        if result.dtype.names is None:
            msg = "There must be some output variables specified for the simulator."
            raise ValueError(msg)

        step = 0
        while simulator.time <= simulator.stop_time:
            step_result = simulator.step()
            for name in result.dtype.names:
                result[step][name] = step_result[name]

            step += 1
        return result

    def reset(self, init_values: Mapping[str, float] | None = None) -> None:
        """Reset FMU to specified initial condition.

        :param init_values: Values for initialization.
        """
        self.time = self.start_time
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=self.start_time)

        # set init values
        # instead of using the fmpy apply_start_values func from fmpy use the own set_values func to set the values
        # of the simulation variables correctly, reasons are also performance and simulation speed
        self.set_values(init_values)  # type: ignore[arg-type]

        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def close(self) -> None:
        """Close the FMU and tidy up the unzipped files."""
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self._unzipdir)  # clean up unzipped files

    @classmethod
    @contextmanager
    def inspect(cls, fmu_path: pathlib.Path | str) -> Iterator[FMUContext]:
        """
        Context manager for inspecting an FMU to extract metadata.

        Initializes a temporary FMU simulation environment to extract information such as
        input/output variable names, parameter start values, and variable bounds. Returns
        this data as a dictionary for use in TOML export or further analysis.

        This method handles FMU loading, parsing, and cleanup internally, and is safe to use
        with `with` blocks. If initialization fails, it yields `None`.

        :param fmu_path: Path to the FMU file (.fmu) as a string or Path.
        :param output_path: Optional output path for the resulting file or reference.
        :yield: A dictionary with FMU metadata, or None on failure.
        """

        def parse_numeric(value: str | None) -> float | None:
            if value is None:
                return None
            try:
                number = float(value)
                return int(number) if number.is_integer() else number
            except (ValueError, TypeError):
                return None

        fmu_path = pathlib.Path(fmu_path)
        simulator = cls(
            _id=0,
            fmu_path=fmu_path,
            start_time=0.0,
            stop_time=1.0,
            step_size=0.1,
        )
        model_description = simulator.model_description

        try:
            actions: list[VariableDict] = []
            observations: list[VariableDict] = []
            parameters: dict[str, str | None] = {}
            var_dict: VariableDict
            for var in model_description.modelVariables:
                if "." in var.name or "[" in var.name or "]" in var.name:
                    continue

                if var.causality == "input":
                    var_dict = {"name": var.name, "is_ext_input": True}
                    min_val = parse_numeric(getattr(var, "min", None))
                    max_val = parse_numeric(getattr(var, "max", None))
                    if min_val is not None:
                        var_dict["low_value"] = min_val
                    if max_val is not None:
                        var_dict["high_value"] = max_val
                    actions.append(var_dict)

                elif var.causality == "output":
                    var_dict = {"name": var.name, "is_ext_output": True}
                    min_val = parse_numeric(getattr(var, "min", None))
                    max_val = parse_numeric(getattr(var, "max", None))
                    if min_val is not None:
                        var_dict["low_value"] = min_val
                    if max_val is not None:
                        var_dict["high_value"] = max_val
                    observations.append(var_dict)

                elif var.causality == "parameter":
                    parameters[var.name] = str(getattr(var, "start", None)) if hasattr(var, "start") else None
            yield {
                "fmu_name": fmu_path.stem,
                "fmu_path": fmu_path,
                "actions": actions,
                "observations": observations,
                "parameters": parameters,
            }

        except Exception:
            log.exception("Failed to inspect FMU simulator")

        finally:
            if simulator:
                simulator.close()


class VariableDict(TypedDict, total=False):
    name: str
    is_ext_input: bool
    is_ext_output: bool
    low_value: Number
    high_value: Number


class FMUContext(TypedDict):
    fmu_name: str
    fmu_path: pathlib.Path
    actions: list[VariableDict]
    observations: list[VariableDict]
    parameters: dict[str, str | None]


class FMU2MESlave(FMU2Model):
    """Helper class for simulation of FMU2 FMUs. This is as wrapper for FMU2Model.
    It can be used to wrap model exchange FMUs such that they can be simulated similar to a co-simulation FMU. This
    is especially helpful for testing model exchange FMUs.

    It exposes an interface that emulates part of the original FMU2Slave class from fmpy.
    """

    # Define some constants that might be needed according to the FMI Standard
    fmi2True: int = 1  # noqa: N815
    fmi2False: int = 0  # noqa: N815

    fmi2OK: int = 0  # noqa: N815
    fmi2Warning: int = 1  # noqa: N815
    fmi2Discard: int = 2  # noqa: N815
    fmi2Error: int = 3  # noqa: N815
    fmi2Fatal: int = 4  # noqa: N815
    fmi2Pending: int = 5  # noqa: N815

    def __init__(self, **kwargs: Any) -> None:
        r"""Initialize the FMU2Slave object. See also the fmyp documentation :py:class:`fmpy.fmi2.FMU2Model`.

        :param Any \**kwargs: Accepts any parameters that fmpy.FMU2Model accepts.
        """
        super().__init__(**kwargs)
        self._model_description: ModelDescription = read_model_description(kwargs["unzipDirectory"])
        self._solver: CVodeSolver
        self._tolerance: float = 0.0
        self._stop_time: float = 0.0
        self._start_time: float = 0.0

    def setupExperiment(  # noqa: N802
        self,
        tolerance: float | None = None,
        startTime: float = 0.0,  # noqa:N803
        stopTime: float | None = None,  # noqa:N803
        **kwargs: Any,
    ) -> int:
        """Experiment setup and storage of required values.

        .. see also::
            fmpy.fmi2.FMU2Model.setupExperiment

        :param tolerance: Solver tolerance, default value is 1e-5.
        :param startTime: Starting time for the experiment.
        :param stopTime: Ending time for the experiment.
        :param kwargs: Other keyword arguments that might be required for FMU2Model.setupExperiment in the future.
        :return: FMI2 return value.
        """
        self._tolerance = 1e-5 if tolerance is None else tolerance
        self._stop_time = 0.0 if stopTime is None else stopTime
        self._start_time = startTime

        kwargs["tolerance"] = self._tolerance
        kwargs["stopTime"] = self._stop_time
        kwargs["startTime"] = self._start_time

        return super().setupExperiment(**kwargs)

    def exitInitializationMode(self, **kwargs: Any) -> int:  # noqa: N802
        """Exit the initialization mode and set up the cvode solver.

        See also: :py:class:`fmpy.fmi2.FMU2Model.exitInitializationMode`

        :param kwargs: Keyword arguments accepted by FMU2Model.exitInitializationMode.
        :return: FMI2 return value.
        """
        ret = super().exitInitializationMode(**kwargs)

        # Collect discrete states from FMU
        self.eventInfo.newDiscreteStatesNeeded = self.fmi2true
        self.eventInfo.terminateSimulation = self.fmi2false

        while (
            self.eventInfo.newDiscreteStatesNeeded == self.fmi2true
            and self.eventInfo.terminateSimulation == self.fmi2false
        ):
            # update discrete states
            self.newDiscreteStates()
        self.enterContinuousTimeMode()

        # Initialize solver
        self._solver = CVodeSolver(
            set_time=self.setTime,
            startTime=self._start_time,
            maxStep=(self._stop_time - self._start_time) / 50.0,
            relativeTolerance=self._tolerance,
            nx=self._model_description.numberOfContinuousStates,
            nz=self._model_description.numberOfEventIndicators,
            get_x=self.getContinuousStates,
            set_x=self.setContinuousStates,
            get_dx=self.getDerivatives,
            get_z=self.getEventIndicators,
        )

        return ret

    def doStep(  # noqa: N802
        self,
        currentCommunicationPoint: float,  # noqa: N803
        communicationStepSize: float,  # noqa: N803
        noSetFMUStatePriorToCurrentPoint: int | None = None,  # noqa: N803
    ) -> int:
        """Perform a simulation step. Advance simulation from *currentCommunicationPoint* by *communicationStepSize*.

        Also refer to the FMI2 Standard documentation.

        :param currentCommunicationPoint: Current time stamp (starting point for simulation step).
        :param communicationStepSize: Time step size.
        :param noSetFMUStatePriorToCurrentPoint: Determine whether a reset before *currentCommunicationPoint* is
            possible. Must be either fmi2True or fmi2False.
        :return: FMU2 return value.
        """
        time = currentCommunicationPoint
        step_size = communicationStepSize

        # Perform a solver step and reset the FMU Model time.
        _, time = self._solver.step(time, time + step_size)
        self.setTime(time)
        # Check for events that might have occurred during the step
        step_event, _ = self.completedIntegratorStep()

        return self.fmi2ok
