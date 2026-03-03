"""Tests for __str__ and __repr__ of FMUSimulator."""

import pytest

from eta_ctrl.simulators import FMUSimulator


class TestFMUSimulatorStringRepresentations:
    """Tests for FMUSimulator.__str__ and __repr__."""

    @pytest.fixture(scope="class")
    def named_simulator(self, config_fmu):
        """FMUSimulator with explicit input/output name lists."""
        return FMUSimulator(
            0,
            fmu_path=config_fmu["file"],
            start_time=0,
            stop_time=100,
            step_size=1,
            names_inputs=["u"],
            names_outputs=["s", "v", "a"],
            init_values={"u": 0},
        )

    @pytest.fixture(scope="class")
    def all_vars_simulator(self, config_fmu):
        """FMUSimulator with all model variables for both inputs and outputs."""
        return FMUSimulator(0, fmu_path=config_fmu["file"])

    # --- __str__ ---

    def test_str_exact_for_named_simulator(self, named_simulator):
        expected = "FMUSimulator('damped_oscillator', 1 inputs, 3 outputs)"
        assert str(named_simulator) == expected

    def test_str_contains_fmu_stem(self, named_simulator):
        assert "damped_oscillator" in str(named_simulator)

    def test_str_format_for_all_vars_simulator(self, all_vars_simulator):
        """Verify str format is correct; counts reflect all available model variables."""
        n_inputs = len(all_vars_simulator.input_mapping)
        n_outputs = len(all_vars_simulator.output_mapping)
        expected = f"FMUSimulator('damped_oscillator', {n_inputs} inputs, {n_outputs} outputs)"
        assert str(all_vars_simulator) == expected

    def test_str_input_output_counts_match_mappings(self, named_simulator):
        """Verify str counts are consistent with actual input/output mappings."""
        result = str(named_simulator)
        assert f"{len(named_simulator.input_mapping)} inputs" in result
        assert f"{len(named_simulator.output_mapping)} outputs" in result

    # --- __repr__ ---

    def test_repr_contains_fmu_path(self, named_simulator):
        assert "fmu_path=" in repr(named_simulator)
        assert "damped_oscillator" in repr(named_simulator)

    def test_repr_contains_timing_params(self, named_simulator):
        result = repr(named_simulator)
        assert "start_time=0" in result
        assert "stop_time=100" in result
        assert "step_size=1" in result

    def test_repr_timing_reflects_constructor_args(self, config_fmu):
        """Verify repr timing fields match the values passed to the constructor."""
        sim = FMUSimulator(
            1,
            fmu_path=config_fmu["file"],
            start_time=10,
            stop_time=500,
            step_size=5,
        )
        result = repr(sim)
        assert "start_time=10" in result
        assert "stop_time=500" in result
        assert "step_size=5" in result
