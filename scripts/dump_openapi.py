"""Print the engine's OpenAPI document (stable formatting) to stdout."""

import json

from kibble_engine.api.app import create_app

if __name__ == "__main__":
    print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))
