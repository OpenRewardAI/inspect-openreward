from __future__ import annotations

from types import SimpleNamespace

from inspect_ai.tool import ToolDef

from inspect_openreward._task import _RewardTracker, _blocks_to_input, _make_async_tool


# -- Stubs -------------------------------------------------------------------


class _StubSession:
    """Fake async session that records calls and returns canned responses."""

    def __init__(self, blocks, reward=None, finished=False):
        self._blocks = blocks
        self._reward = reward
        self._finished = finished
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, kwargs):
        self.calls.append((name, kwargs))
        return SimpleNamespace(
            blocks=self._blocks,
            reward=self._reward,
            finished=self._finished,
        )


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


# -- _make_async_tool --------------------------------------------------------


async def test_tool_metadata_round_trips():
    tracker = _RewardTracker()
    session = _StubSession(blocks=[SimpleNamespace(text="hi")])
    tool = _make_async_tool(_spec(), session, tracker)

    td = ToolDef(tool)
    assert td.name == "echo"
    assert td.description == "Echo a message."
    assert "message" in td.parameters.properties


async def test_tool_forwards_call_and_returns_text():
    tracker = _RewardTracker()
    session = _StubSession(blocks=[SimpleNamespace(text="hello world")])
    tool = _make_async_tool(_spec(), session, tracker)

    result = await ToolDef(tool).tool(message="hello world")

    assert result == "hello world"
    assert session.calls == [("echo", {"message": "hello world"})]


async def test_empty_blocks_returns_empty_string():
    tracker = _RewardTracker()
    session = _StubSession(blocks=[])
    tool = _make_async_tool(_spec(), session, tracker)

    result = await ToolDef(tool).tool(message="x")

    assert result == ""


async def test_reward_tracked():
    tracker = _RewardTracker()
    session = _StubSession(blocks=[SimpleNamespace(text="ok")], reward=0.75)
    tool = _make_async_tool(_spec(), session, tracker)

    await ToolDef(tool).tool(message="x")

    assert tracker.reward == 0.75
    assert not tracker.finished


async def test_finished_appends_marker():
    tracker = _RewardTracker()
    session = _StubSession(
        blocks=[SimpleNamespace(text="done")], reward=1.0, finished=True
    )
    tool = _make_async_tool(_spec(), session, tracker)

    result = await ToolDef(tool).tool(message="x")

    assert "[Episode complete]" in result
    assert tracker.finished


async def test_finished_empty_blocks():
    tracker = _RewardTracker()
    session = _StubSession(blocks=[], reward=1.0, finished=True)
    tool = _make_async_tool(_spec(), session, tracker)

    result = await ToolDef(tool).tool(message="x")

    assert result == "[Episode complete]"


# -- _blocks_to_input --------------------------------------------------------


def test_single_text_block_returns_string():
    block = SimpleNamespace(text="hello")
    # TextBlock check uses isinstance, so we need proper types
    from openreward.api.environments.types import TextBlock

    result = _blocks_to_input([TextBlock(text="hello")])
    assert result == "hello"


def test_empty_blocks_returns_empty():
    result = _blocks_to_input([])
    assert result == ""


# -- _RewardTracker ----------------------------------------------------------


def test_tracker_ignores_none_reward():
    tracker = _RewardTracker()
    tracker.update(None, False)
    assert tracker.reward == 0.0


def test_tracker_keeps_last_reward():
    tracker = _RewardTracker()
    tracker.update(0.5, False)
    tracker.update(0.8, False)
    assert tracker.reward == 0.8


def test_tracker_finished_is_sticky():
    tracker = _RewardTracker()
    tracker.update(1.0, True)
    tracker.update(None, False)
    assert tracker.finished
