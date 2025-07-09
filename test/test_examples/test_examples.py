import logging
import pathlib
import shutil

import pytest

from eta_ctrl.envs.sim_env import SimEnv
from examples.damped_oscillator.main import (
    experiment_conventional as ex_oscillator,
    get_path as get_oscillator_path,
)
from examples.pendulum.main import (
    conventional as ex_pendulum_conventional,
    get_path as get_pendulum_path,
    machine_learning as ex_pendulum_learning,
)


class TestPendulumExample:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_pendulum_path()
        yield path
        shutil.rmtree(path / "results")

    def test_conventional(self, experiment_path):
        ex_pendulum_conventional(
            experiment_path,
            {
                "settings": {"log_to_file": False},
                "environment_specific": {"do_render": False},
            },
        )

    def test_learning(self, experiment_path):
        ex_pendulum_learning(
            experiment_path,
            {
                "settings": {
                    "n_episodes_learn": 2,
                    "save_model_every_x_episodes": 2,
                    "n_environments": 1,
                    "log_to_file": False,
                },
                "environment_specific": {"do_render": False},
            },
        )


class TestOscillatorExample:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_oscillator_path()
        yield path
        logging.shutdown()
        shutil.rmtree(path / "results")

    def test_oscillator(self, experiment_path):
        ex_oscillator(experiment_path, {"settings": {"log_to_file": False}})


class TestFMUExport:
    """Test the FMU variables export functionality."""

    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_oscillator_path()
        yield path
        # Clean up any test files created - including numbered variants due to overwrite protection

        # Clean up files in the experiment path
        for toml_file in path.glob("damped_oscillator_structure*.toml"):
            toml_file.unlink(missing_ok=True)

        # Clean up files in current directory
        for toml_file in pathlib.Path().glob("damped_oscillator_structure*.toml"):
            toml_file.unlink(missing_ok=True)

        # Clean up any other test files
        for toml_file in pathlib.Path().glob("test_structure*.toml"):
            toml_file.unlink(missing_ok=True)

    @pytest.fixture(scope="class")
    def fmu_path(self, experiment_path):
        """Get the FMU file path for testing."""
        return experiment_path / "damped_oscillator.fmu"

    def test_fmu_file_exists(self, fmu_path):
        """Test that the FMU file exists before testing export functionality."""
        assert fmu_path.exists(), f"FMU file not found at {fmu_path}"

    def test_export_to_default_location(self, fmu_path):
        """Test export to default location (same directory as FMU)."""
        SimEnv.export_fmu_structure(fmu_path)

        # Check if default file was created
        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"
        assert default_path.exists(), f"Default export file not created at {default_path}"

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_export_to_custom_location(self, fmu_path):
        """Test export to custom location with full file path."""
        custom_path = pathlib.Path("./damped_oscillator_structure_custom.toml")
        SimEnv.export_fmu_structure(fmu_path, custom_path)

        # Check if custom file was created
        assert custom_path.exists(), f"Custom export file not created at {custom_path}"

        # Clean up the file after test
        if custom_path.exists():
            custom_path.unlink()

    def test_toml_file_structure(self, fmu_path):
        """Test that the exported TOML file has the expected structure."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        # Check expected structure for new format
        assert "fmu_info" in toml_data
        assert "actions" in toml_data
        assert "observations" in toml_data
        assert isinstance(toml_data["actions"], list)
        assert isinstance(toml_data["observations"], list)

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_fmu_info_section(self, fmu_path):
        """Test that the fmu_info section contains expected fields."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        # Check that fmu_info section exists and has expected fields
        assert "fmu_info" in toml_data
        fmu_info = toml_data["fmu_info"]
        assert "name" in fmu_info
        assert "path" in fmu_info
        assert "description" in fmu_info
        assert fmu_info["name"] == fmu_path.stem

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_has_model_actions(self, fmu_path):
        """Test that at least one model action exists."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        assert len(toml_data["actions"]) > 0, "FMU should have at least one model action"

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_has_model_observations(self, fmu_path):
        """Test that at least one model observation exists."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        assert len(toml_data["observations"]) > 0, "FMU should have at least one model observation"

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_model_actions_structure(self, fmu_path):
        """Test that model actions have the expected structure."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        # Check that actions have correct structure
        for action in toml_data["actions"]:
            assert "name" in action
            assert isinstance(action["name"], str)
            assert len(action["name"]) > 0, "Action name should not be empty"

            # Should only contain name, low_value, high_value
            for key in action.keys():
                assert key in ["name", "is_ext_input", "low_value", "high_value"], (
                    f"Unexpected key '{key}' in model action"
                )

            # low_value and high_value should be numbers (int, float) or numeric strings, or None
            if "low_value" in action and action["low_value"] is not None:
                if isinstance(action["low_value"], str):
                    # If it's a string, it should be convertible to float
                    try:
                        float(action["low_value"])
                    except ValueError:
                        pytest.fail(f"low_value '{action['low_value']}' is not a valid number")
                else:
                    assert isinstance(action["low_value"], (int, float))
            if "high_value" in action and action["high_value"] is not None:
                if isinstance(action["high_value"], str):
                    # If it's a string, it should be convertible to float
                    try:
                        float(action["high_value"])
                    except ValueError:
                        pytest.fail(f"high_value '{action['high_value']}' is not a valid number")
                else:
                    assert isinstance(action["high_value"], (int, float))

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_model_observations_structure(self, fmu_path):
        """Test that model observations have the expected structure."""
        SimEnv.export_fmu_structure(fmu_path)

        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"

        import toml

        with pathlib.Path.open(default_path) as f:
            toml_data = toml.load(f)

        # Check that observations have correct structure
        for observation in toml_data["observations"]:
            assert "name" in observation
            assert isinstance(observation["name"], str)
            assert len(observation["name"]) > 0, "Observation name should not be empty"

            # Should only contain name, low_value, high_value
            for key in observation.keys():
                assert key in ["name", "is_ext_output", "low_value", "high_value"], (
                    f"Unexpected key '{key}' in model observation"
                )

            # low_value and high_value should be numbers (int, float) or numeric strings, or None
            if "low_value" in observation and observation["low_value"] is not None:
                if isinstance(observation["low_value"], str):
                    # If it's a string, it should be convertible to float
                    try:
                        float(observation["low_value"])
                    except ValueError:
                        pytest.fail(f"low_value '{observation['low_value']}' is not a valid number")
                else:
                    assert isinstance(observation["low_value"], (int, float))
            if "high_value" in observation and observation["high_value"] is not None:
                if isinstance(observation["high_value"], str):
                    # If it's a string, it should be convertible to float
                    try:
                        float(observation["high_value"])
                    except ValueError:
                        pytest.fail(f"high_value '{observation['high_value']}' is not a valid number")
                else:
                    assert isinstance(observation["high_value"], (int, float))

        # Clean up the file after test
        if default_path.exists():
            default_path.unlink()

    def test_overwrite_protection(self, fmu_path):
        """Test that overwrite protection creates unique filenames."""
        # Create first file
        SimEnv.export_fmu_structure(fmu_path)
        default_path = fmu_path.parent / f"{fmu_path.stem}_structure.toml"
        assert default_path.exists()

        # Create second file - should have _1 suffix due to overwrite protection
        SimEnv.export_fmu_structure(fmu_path)
        protected_path = fmu_path.parent / f"{fmu_path.stem}_structure_1.toml"
        assert protected_path.exists()

        # Clean up both files
        if default_path.exists():
            default_path.unlink()
        if protected_path.exists():
            protected_path.unlink()
