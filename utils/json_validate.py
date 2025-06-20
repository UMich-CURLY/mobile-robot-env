import json
import jsonschema
from jsonschema import validate

# Define the JSON schema
schema = {
    "type": "object",
    "properties": {
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {
                    "type": "object",
                    "properties": {
                        "ground_json": {
                            "type": "object",
                            "properties": {
                                "target": {
                                    "type": "object",
                                    "properties": {
                                        "phrase": {"type": "string"}
                                    },
                                    "required": ["phrase"]
                                },
                                "landmark": {
                                    "type": "object",
                                    "properties": {
                                        "phrase": {"type": "string"},
                                        "relation to target": {"type": "string"}
                                    },
                                    "required": ["phrase", "relation to target"]
                                }
                            },
                            "required": ["target", "landmark"]
                        }
                    },
                    "required": ["ground_json"]
                }
            },
            "required": ["name", "args"]
        }
    },
    "required": ["command"]
}