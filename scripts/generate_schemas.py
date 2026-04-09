import json
import math
from pathlib import Path

from eta_ctrl.config.config import Config
from eta_ctrl.envs.state import StateStructure


def main() -> None:
    schema_path = Path(".vscode/schemas")

    # StateConfig
    state_config_schema = create_state_config_schema()
    export_schema(path=schema_path / "state_config.json", schema=state_config_schema)

    # Config
    config_schema = Config.model_json_schema()
    export_schema(path=schema_path / "config.json", schema=config_schema)


def create_state_config_schema() -> dict:
    schema = StateStructure.model_json_schema()

    def modify_number(schema: list | dict) -> None:
        if isinstance(schema, dict):
            for key, value in schema.items():
                if isinstance(value, (list, dict)):
                    modify_number(value)
                # Allow strings for number and integer fields
                elif key == "type" and value in ("number", "integer"):
                    schema[key] = [value, "string"]
                # Replace Infinity with a string representation
                elif key == "default" and math.isinf(value or 0):
                    schema[key] = str(value)

        elif isinstance(schema, list):
            for item in schema:
                modify_number(item)

    modify_number(schema=schema)
    return schema


def export_schema(path: Path, schema: dict) -> None:
    """Helper function."""
    with path.open("w") as f:
        json.dump(schema, f, indent=4)


if __name__ == "__main__":
    main()
