import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

import eta_ctrl
from eta_ctrl.config.config import Config
from eta_ctrl.util.io_utils import load_config

REPO_ROOT = Path(eta_ctrl.__file__).parent.parent

CONFIG_FILES = [
    REPO_ROOT / "test" / "resources" / "config" / "config1.json",
    REPO_ROOT / "test" / "resources" / "config" / "config2.toml",
    REPO_ROOT / "test" / "resources" / "config" / "config3.yaml",
    REPO_ROOT / "examples" / "kea_tank" / "config.toml",
    REPO_ROOT / "examples" / "damped_oscillator" / "config_learning.json",
    REPO_ROOT / "examples" / "damped_oscillator" / "config_conventional.json",
    REPO_ROOT / "examples" / "pendulum" / "config_learning.json",
    REPO_ROOT / "examples" / "pendulum" / "config_conventional.json",
]


class TestConfigSchema:
    def test_config_schema_generation(self):
        # Will also produce a schema for all other included classes (ConfigSetup, ...)
        schema = Config.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema


class TestConfigFileValidation:
    @pytest.fixture(scope="class")
    def validator(self, root_path):
        schema_path = root_path / ".vscode/schemas/config.json"
        return Draft7Validator(json.loads(schema_path.read_text()))

    @pytest.mark.parametrize("config_file", CONFIG_FILES, ids=lambda p: p.name)
    def test_settings_valid(self, config_file: Path, validator):
        """Validate that each config file's settings section conforms to the JSON schema."""
        data = load_config(config_file)
        errors = list(validator.iter_errors(data))
        assert errors == [], "\n".join(str(e) for e in errors)
