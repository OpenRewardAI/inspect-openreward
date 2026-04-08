from __future__ import annotations

from types import SimpleNamespace

from inspect_ai.tool._tool_def import ToolDef

from inspect_openreward import openreward_tool_to_inspect


class _StubSession:
    def __init__(self, blocks):
        self._blocks = blocks
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, name, kwargs):
        self.calls.append((name, kwargs))
        return SimpleNamespace(blocks=self._blocks)


def _spec(**overrides):
    return SimpleNamespace(
        name="echo",
        description="Echo a message.",
        input_schema={
            "properties": {
                "message": {"type": "string", "description": "text"},
            },
            "required": ["message"],
        },
        **overrides,
    )


async def test_round_trip_metadata():
    session = _StubSession(blocks=[SimpleNamespace(text="hi")])
    tool = openreward_tool_to_inspect(_spec(), session)

    td = ToolDef(tool)
    assert td.name == "echo"
    assert td.description == "Echo a message."
    assert "message" in td.parameters.properties


async def test_call_forwards_and_returns_text():
    session = _StubSession(blocks=[SimpleNamespace(text="hello world")])
    tool = openreward_tool_to_inspect(_spec(), session)

    result = await ToolDef(tool).tool(message="hello world")

    assert result == "hello world"
    assert session.calls == [("echo", {"message": "hello world"})]


async def test_empty_blocks_returns_empty_string():
    session = _StubSession(blocks=[])
    tool = openreward_tool_to_inspect(_spec(), session)

    result = await ToolDef(tool).tool(message="x")

    assert result == ""
