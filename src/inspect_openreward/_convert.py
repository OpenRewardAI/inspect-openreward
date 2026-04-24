from __future__ import annotations

from typing import Any, Optional

from inspect_ai.tool._tool import Tool
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams
from openreward import Provider, ToolSpec, sanitize_tool_schema
from openreward.api.environments.client import Session


def openreward_tool_to_inspect(
    tool_spec: ToolSpec,
    session: Session,
    provider: Optional[Provider] = None,
) -> Tool:
    """Convert an OpenReward ToolSpec into an Inspect AI Tool.

    ``provider`` selects provider-specific sanitization of the tool's JSON
    schema via ``openreward.sanitize_tool_schema``. Pass ``None`` to leave the
    schema untouched.
    """
    name = tool_spec.name

    async def execute(**kwargs: Any) -> str:
        result = session.call_tool(name, kwargs)
        return result.blocks[0].text if result.blocks else ""

    parameters = (
        dict(tool_spec.input_schema or {})
        if provider is None
        else sanitize_tool_schema(tool_spec.input_schema, provider)
    )

    return ToolDef(
        tool=execute,
        name=name,
        description=tool_spec.description,
        parameters=ToolParams(**parameters),
    ).as_tool()
