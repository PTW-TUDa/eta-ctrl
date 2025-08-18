import json
import math
from pathlib import Path

from eta_ctrl.envs.state import StateStructure


def export_structure_schema() -> None:
    path = Path(".vscode/schemas/structure.json")
    schema = StateStructure.model_json_schema()

    def modify_number(schema: list | dict) -> None:
        if isinstance(schema, dict):
            for key, value in schema.items():
                if isinstance(value, (list, dict)):
                    modify_number(value)
                # Allow strings for number fields
                elif key == "type" and value == "number":
                    schema[key] = ["number", "string"]
                # Replace Infinity with a string representation
                elif key == "default" and math.isinf(value):
                    schema[key] = str(value)

        elif isinstance(schema, list):
            for item in schema:
                modify_number(item)

    modify_number(schema=schema)
    with path.open("w") as f:
        json.dump(schema, f)


if __name__ == "__main__":
    export_structure_schema()
