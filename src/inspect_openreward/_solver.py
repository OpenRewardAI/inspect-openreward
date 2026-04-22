from __future__ import annotations

from typing import Any, Optional

from inspect_ai._util.content import ContentImage, ContentText
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolChoice, ToolDef
from inspect_ai.tool._tool_params import ToolParams
from openreward import (
    BuiltinToolset,
    Provider,
    ToolSpec,
    sanitize_tool_schema,
)
from openreward.api.environments.client import Environment
from openreward.api.environments.types import ImageBlock, TextBlock

from ._constants import (
    FINISHED_METADATA_KEY,
    REWARD_METADATA_KEY,
    TASK_METADATA_KEY,
)

_INSPECT_API_TO_OR_PROVIDER: dict[str, Provider] = {
    "openai": "openai",
    "openai-api": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "vertex": "google",
    "openrouter": "openrouter",
}


@solver
def openreward_solver(
    environment: Environment,
    toolset: Optional[BuiltinToolset] = None,
    tool_choice: ToolChoice = "auto",
) -> Solver:
    """Solver that runs an Inspect agent against an OpenReward session.

    Per sample, this solver:
      1. Opens an `environment.session(task=..., toolset=toolset)`.
      2. Fetches `session.get_prompt()` and appends it as a user message.
      3. Converts `session.list_tools()` into Inspect tools, auto-detecting
         the JSON-schema sanitisation format from the model provider.
      4. Delegates to `generate(tool_calls="loop")` to run the react-style loop.
      5. Captures the terminal `reward` / `finished` from tool outputs into
         `state.metadata`, where `openreward_scorer` can read them.
      6. Closes the session on teardown.

    Args:
        environment: The OpenReward `Environment` to open sessions against.
            The dataset should be built from the same environment.
        toolset: Optional toolset name to pass to `environment.session(...)`.
            E.g. `"claude-code"` to expose a harness-native bash tool surface
            instead of the environment's own tools.
        tool_choice: Passed through to Inspect's `state.tool_choice`.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task = (state.metadata or {}).get(TASK_METADATA_KEY)
        if task is None:
            raise ValueError(
                f"Sample {state.sample_id!r} is missing metadata "
                f"{TASK_METADATA_KEY!r} — use openreward_dataset() to build "
                "samples for this solver."
            )

        provider_format: Provider = _INSPECT_API_TO_OR_PROVIDER[state.model.api] if state.model.api in _INSPECT_API_TO_OR_PROVIDER else "openai"

        with environment.session(task=task, toolset=toolset) as session:
            prompt_blocks = session.get_prompt()
            prompt_content = _blocks_to_content(prompt_blocks)
            if state.messages and isinstance(state.messages[0], ChatMessageUser):
                state.messages[0].content = prompt_content
            else:
                state.messages.insert(0, ChatMessageUser(content=prompt_content))

            state.tools = [
                _wrap_tool(spec, session, provider_format, state)
                for spec in session.list_tools()
            ]
            state.tool_choice = tool_choice

            state = await generate(state, tool_calls="loop")

        return state

    return solve


def _blocks_to_content(
    blocks: list[TextBlock | ImageBlock],
) -> list[ContentText | ContentImage]:
    content: list[ContentText | ContentImage] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            content.append(ContentText(text=block.text))
        elif isinstance(block, ImageBlock):
            data_uri = f"data:{block.mimeType};base64,{block.data}"
            content.append(ContentImage(image=data_uri))
    return content


def _wrap_tool(
    tool_spec: ToolSpec,
    session: Any,
    provider_format: Provider,
    state: TaskState,
) -> Tool:
    name = tool_spec.name

    async def execute(**kwargs: Any) -> str:
        result = session.call_tool(name, kwargs)
        if result.reward is not None:
            state.metadata[REWARD_METADATA_KEY] = result.reward
        if result.finished:
            state.metadata[FINISHED_METADATA_KEY] = True
        return "".join(
            block.text for block in result.blocks if isinstance(block, TextBlock)
        )

    return ToolDef(
        tool=execute,
        name=name,
        description=tool_spec.description,
        parameters=ToolParams(
            **sanitize_tool_schema(tool_spec.input_schema, provider_format)
        ),
    ).as_tool()
