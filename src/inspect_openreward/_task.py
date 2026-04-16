"""Inspect AI task factory for OpenReward environments.

Provides a high-level ``openreward()`` task factory (analogous to
``harbor()`` in inspect_harbor) that fetches tasks from a named OpenReward
environment and evaluates an agent against each one.

Layer structure
---------------
* **Low-level** – :func:`inspect_openreward.openreward_tool_to_inspect`:
  converts a single OR ``ToolSpec`` + ``Session`` into an Inspect ``Tool``.
* **High-level** (this module) – :func:`openreward`, :func:`openreward_solver`,
  :func:`openreward_scorer`: manage the full task/environment lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import Agent, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool._tool import Tool
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.tool._tool_params import ToolParams

from openreward import AsyncOpenReward, OpenReward
from openreward.api.environments.types import (
    ImageBlock,
    Task as ORTask,
    TextBlock,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _RewardTracker:
    """Accumulates the reward signal emitted across tool calls in a session.

    OpenReward returns a ``reward`` float and a ``finished`` flag on every
    ``call_tool`` response.  We keep the *last* non-None reward value, which
    is the standard pattern for environments that only signal reward at the
    end of an episode, and record whether any call marked the episode as done.
    """

    def __init__(self) -> None:
        self.reward: float = 0.0
        self.finished: bool = False

    def update(self, reward: float | None, finished: bool) -> None:
        if reward is not None:
            self.reward = reward
        if finished:
            self.finished = True


def _make_async_tool(
    tool_spec: ToolSpec,
    session: Any,  # openreward.api.environments.client.AsyncSession
    tracker: _RewardTracker,
) -> Tool:
    """Create an Inspect Tool backed by an async OR session.

    Wraps ``session.call_tool`` so that every invocation updates *tracker*
    with the latest reward and finished state.  When the environment signals
    episode completion (``finished=True``), the tool appends
    ``"[Episode complete]"`` to its return value so the model knows to submit.
    """
    name = tool_spec.name

    async def execute(**kwargs: Any) -> str:
        result = await session.call_tool(name, kwargs)
        tracker.update(result.reward, result.finished)
        text = result.blocks[0].text if result.blocks else ""
        if result.finished:
            return f"{text}\n\n[Episode complete]" if text else "[Episode complete]"
        return text

    return ToolDef(
        tool=execute,
        name=name,
        description=tool_spec.description,
        parameters=ToolParams(**(tool_spec.input_schema or {})),
    ).as_tool()


def _blocks_to_input(
    blocks: list[TextBlock | ImageBlock],
) -> str | list[ContentText | ContentImage]:
    """Convert OR prompt blocks to an Inspect message content value."""
    if not blocks:
        return ""
    if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
        return blocks[0].text
    content: list[ContentText | ContentImage] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            content.append(ContentText(text=block.text))
        elif isinstance(block, ImageBlock):
            content.append(
                ContentImage(image=f"data:{block.mimeType};base64,{block.data}")
            )
    return content


def _or_task_to_sample(or_task: ORTask, idx: int) -> Sample:
    """Convert an OpenReward Task to an Inspect Sample.

    The real prompt is fetched lazily by :func:`openreward_solver` when the
    session opens, so the Sample input is a lightweight placeholder.  Task
    routing information is stored in ``metadata`` for the solver to
    reconstruct the ``ORTask``.
    """
    return Sample(
        input=f"OpenReward task {idx} ({or_task.environment_name})",
        id=str(idx),
        metadata={
            "or_server_name": or_task.server_name,
            "or_environment_name": or_task.environment_name,
            "or_task_spec": dict(or_task.task_spec),
            "or_namespace": or_task.namespace,
        },
    )


def _load_samples(
    environment_name: str,
    split: str,
    n_tasks: int | None,
) -> list[Sample]:
    """Fetch OR tasks at task-definition time and convert to Inspect Samples.

    Uses the synchronous ``OpenReward`` client so this can be called from
    ordinary (non-async) task-factory code.
    """
    with OpenReward() as client:
        env = client.environments.get(name=environment_name)
        or_tasks = env.list_tasks(split=split)

    if n_tasks is not None:
        or_tasks = or_tasks[:n_tasks]

    return [_or_task_to_sample(t, idx) for idx, t in enumerate(or_tasks)]


# ---------------------------------------------------------------------------
# Public solver / scorer / task
# ---------------------------------------------------------------------------


@solver
def openreward_solver(
    agent: Callable[[list[Tool]], Agent] | None = None,
) -> Solver:
    """Solver that manages an OpenReward session and runs an agent.

    For each sample the solver:

    1. Reconstructs the ``ORTask`` from ``state.metadata``.
    2. Opens an async OR session for that task.
    3. Fetches the real prompt and replaces the placeholder in ``state.messages``.
    4. Converts every OR tool into an Inspect tool that also updates a
       :class:`_RewardTracker`.
    5. Calls *agent* with the session's tools to obtain an :class:`~inspect_ai.agent.Agent`,
       then runs it via :func:`~inspect_ai.agent.as_solver`.
    6. After the session closes, writes ``or_reward`` and ``or_finished`` into
       ``state.metadata`` for the scorer to consume.

    A single ``AsyncOpenReward`` client is created per solver instance and
    shared across all samples, allowing connection re-use.

    Args:
        agent: A callable that receives the list of OR-backed
            :class:`~inspect_ai.tool.Tool` objects and returns an
            :class:`~inspect_ai.agent.Agent`.  Defaults to
            ``lambda tools: react(tools=tools)``.  Use this to customise the
            agent — for example to set ``max_attempts``, add system prompts, or
            swap in a different agent implementation entirely::

                openreward_solver(agent=lambda tools: react(tools=tools, max_attempts=10))
    """
    agent_factory: Callable[[list[Tool]], Agent] = agent or (
        lambda tools: react(tools=tools)
    )
    client = AsyncOpenReward()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        or_task = ORTask(
            server_name=meta["or_server_name"],
            environment_name=meta["or_environment_name"],
            task_spec=meta["or_task_spec"],
            namespace=meta["or_namespace"],
        )

        env = client.environments.get(name=or_task.deployment_name)
        tracker = _RewardTracker()

        async with env.session(task=or_task) as session:
            # Replace placeholder input with the real task prompt
            prompt_blocks = await session.get_prompt()
            state.messages = [ChatMessageUser(content=_blocks_to_input(prompt_blocks))]

            # Build Inspect tools, wired to this session and reward tracker
            tool_specs = await session.list_tools()
            tools = [_make_async_tool(spec, session, tracker) for spec in tool_specs]

            state = await as_solver(agent_factory(tools))(state, generate)

        state.metadata["or_reward"] = tracker.reward
        state.metadata["or_finished"] = tracker.finished
        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def openreward_scorer() -> Scorer:
    """Scorer that reads the reward signal stored by :func:`openreward_solver`.

    Returns a :class:`~inspect_ai.scorer.Score` whose ``value`` is the final
    reward emitted by the OpenReward environment during the agent's session.
    """

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        reward = float(state.metadata.get("or_reward", 0.0))
        finished = bool(state.metadata.get("or_finished", False))
        return Score(
            value=reward,
            answer="FINISHED" if finished else "UNFINISHED",
            explanation=(
                f"OpenReward reward: {reward}. "
                f"Session finished: {finished}."
            ),
        )

    return score


@task
def openreward(
    environment: str,
    split: str = "train",
    n_tasks: int | None = None,
    agent: Callable[[list[Tool]], Agent] | None = None,
) -> Task:
    """Inspect Task backed by an OpenReward environment.

    Fetches tasks from the named OpenReward environment and evaluates an agent
    against each one, using the environment's tools and reward signal.

    Args:
        environment: OpenReward environment name,
            e.g. ``"kanishk/EndlessTerminals"``.
        split: Dataset split to use (default: ``"train"``).
        n_tasks: Maximum number of tasks to include. All tasks if ``None``.
        agent: A callable that receives the list of OR-backed
            :class:`~inspect_ai.tool.Tool` objects and returns an
            :class:`~inspect_ai.agent.Agent`.  Defaults to
            ``lambda tools: react(tools=tools)``.  Use this to customise the
            agent — for example to set ``max_attempts``, add system prompts, or
            swap in a different agent implementation entirely::

                openreward(
                    "kanishk/EndlessTerminals",
                    agent=lambda tools: react(tools=tools, max_attempts=10),
                )

    Returns:
        Configured Inspect AI task.
    """
    samples = _load_samples(environment, split, n_tasks)
    return Task(
        dataset=samples,
        solver=openreward_solver(agent=agent),
        scorer=openreward_scorer(),
    )
