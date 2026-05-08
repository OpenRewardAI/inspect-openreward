from __future__ import annotations

from typing import Any, Optional

from inspect_ai._util.content import ContentImage, ContentText
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.tool import Tool, ToolChoice, ToolDef
from inspect_ai.tool._tool_params import ToolParams
from openreward import (
    AsyncOpenReward,
    BuiltinToolset,
    Provider,
    ToolSpec,
    sanitize_tool_schema,
)
from openreward.api.environments.client import (
    AsyncEnvironment,
    AsyncSession,
    Environment,
)
from openreward.api.environments.types import ImageBlock, TextBlock

from ._constants import (
    FINISHED_METADATA_KEY,
    REWARD_METADATA_KEY,
    TASK_METADATA_KEY,
)
from ._shared import ensure_param_descriptions

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
    solver: Optional[Solver | list[Solver]] = None,
    *,
    toolset: Optional[BuiltinToolset] = None,
    tool_choice: ToolChoice = "auto",
) -> Solver:
    """Wrap an Inspect solver chain with OpenReward session lifecycle management.

    Per sample, this wrapper:
      1. Opens an `environment.session(task=..., toolset=toolset)`.
      2. Fetches `session.get_prompt()` and injects it as the user message.
      3. Converts `session.list_tools()` into Inspect tools (auto-detecting the
         JSON-schema sanitisation format from the model provider) and installs
         them onto `state.tools`, along with `state.tool_choice`.
      4. Runs the caller-supplied inner `solver` (or `generate(tool_calls="loop")`
         by default) inside the open session.
      5. Captures the terminal `reward` / `finished` from tool outputs into
         `state.metadata`, where `openreward_scorer` can read them. This works
         regardless of how the inner chain invokes tools.
      6. Closes the session on teardown.

    The caller passes the synchronous `Environment` (the same one used to build
    the dataset). Internally, the solver lazily constructs an
    `AsyncEnvironment` against the same deployment so that session I/O is fully
    awaitable and does not block Inspect's event loop.

    Args:
        environment: The OpenReward `Environment` to open sessions against.
            The dataset should be built from the same environment.
        solver: Inner solver (or list of solvers, composed via `chain(...)`)
            to run inside the session. Defaults to `generate(tool_calls="loop")`,
            which reproduces the react-style loop. Pass e.g.
            `chain(system_message("..."), generate())` or `basic_agent(...)`
            to plug in different scaffolding. Inner chains can add further
            tools via `use_tools(extra, append=True)`; the session-bound
            tools installed here will remain available.
        toolset: Optional toolset name to pass to `environment.session(...)`.
            E.g. `"claude-code"` to expose a harness-native bash tool surface
            instead of the environment's own tools.
        tool_choice: Passed through to Inspect's `state.tool_choice`.
    """

    inner: Optional[Solver] = (
        chain(*solver) if isinstance(solver, list) else solver
    )

    async_env: Optional[AsyncEnvironment] = None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal async_env

        task = (state.metadata or {}).get(TASK_METADATA_KEY)
        if task is None:
            raise ValueError(
                f"Sample {state.sample_id!r} is missing metadata "
                f"{TASK_METADATA_KEY!r} — use openreward_dataset() to build "
                "samples for this solver."
            )

        provider_format: Provider = _INSPECT_API_TO_OR_PROVIDER[state.model.api] if state.model.api in _INSPECT_API_TO_OR_PROVIDER else "openai"

        if async_env is None:
            async_env = AsyncOpenReward().environments.get(
                environment.deployment_name, variant=environment.variant
            )

        async with async_env.session(task=task, toolset=toolset) as session:
            prompt_blocks = await session.get_prompt()
            prompt_content = _blocks_to_content(prompt_blocks)
            if state.messages and isinstance(state.messages[0], ChatMessageUser):
                state.messages[0].content = prompt_content
            else:
                state.messages.insert(0, ChatMessageUser(content=prompt_content))

            tool_specs = await session.list_tools()
            state.tools = [
                _wrap_tool(spec, session, provider_format, state)
                for spec in tool_specs
            ]
            state.tool_choice = tool_choice

            if inner is None:
                state = await generate(state, tool_calls="loop")
            else:
                state = await inner(state, generate)

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
    session: AsyncSession,
    provider_format: Provider,
    state: TaskState,
) -> Tool:
    name = tool_spec.name

    async def execute(**kwargs: Any) -> str:
        result = await session.call_tool(name, kwargs)
        if result.reward is not None:
            state.metadata[REWARD_METADATA_KEY] = result.reward
        if result.finished:
            state.metadata[FINISHED_METADATA_KEY] = True
        return "".join(
            block.text for block in result.blocks if isinstance(block, TextBlock)
        )

    schema = sanitize_tool_schema(tool_spec.input_schema, provider_format)
    ensure_param_descriptions(schema, name)

    return ToolDef(
        tool=execute,
        name=name,
        description=tool_spec.description,
        parameters=ToolParams(**schema),
    ).as_tool()
