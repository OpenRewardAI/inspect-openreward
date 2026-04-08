from __future__ import annotations

from typing import Any

from inspect_ai.tool._tool import Tool
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams
from openreward.api.environments.client import Session
from openreward.api.environments.types import ToolSpec


def openreward_tool_to_inspect(tool_spec: ToolSpec, session: Session) -> Tool:
    """Convert an OpenReward ToolSpec into an Inspect AI Tool."""
    name = tool_spec.name

    async def execute(**kwargs: Any) -> str:
        result = session.call_tool(name, kwargs)
        return result.blocks[0].text if result.blocks else ""

    return ToolDef(
        tool=execute,
        name=name,
        description=tool_spec.description,
        parameters=ToolParams(**(tool_spec.input_schema or {})),
    ).as_tool()
