from __future__ import annotations

from typing import Any


def ensure_param_descriptions(schema: dict[str, Any], tool_name: str) -> None:
    # Inspect's ToolDef validator rejects any parameter whose `description` is
    # empty, but OpenReward tool schemas (generated from pydantic models) may
    # omit per-field descriptions. Fill in a stub so the transcript can be
    # serialised without changing the model-visible schema semantically.
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for prop_name, prop_schema in properties.items():
        if isinstance(prop_schema, dict) and not prop_schema.get("description"):
            prop_schema["description"] = (
                f"Parameter '{prop_name}' of tool '{tool_name}'."
            )
