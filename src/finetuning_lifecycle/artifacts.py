import json
import os
from typing import Dict, Any

def get_schema_path(schema_name: str) -> str:
    """Resolves the absolute path to a schema file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "schemas", schema_name)

def load_schema(schema_name: str) -> Dict[str, Any]:
    """Loads a JSON schema from the schemas directory."""
    path = get_schema_path(schema_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schema not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def validate_artifact(data: Dict[str, Any], schema_name: str) -> bool:
    """Validates data against a JSON schema. Fallback to basic type-checking if jsonschema isn't installed."""
    try:
        import jsonschema
        schema = load_schema(schema_name)
        jsonschema.validate(instance=data, schema=schema)
        return True
    except ImportError:
        # Fallback basic key verification if jsonschema is not available
        schema = load_schema(schema_name)
        required_keys = schema.get("required", [])
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: '{key}' in artifact.")
        return True
    except Exception as e:
        raise ValueError(f"Schema validation failed: {str(e)}")

def validate_json_file(file_path: str, schema_name: str) -> bool:
    """Validates an existing JSON file against a schema."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact file not found at {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    return validate_artifact(data, schema_name)
