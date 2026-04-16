"""Inspect AI task factory for OpenReward environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import Agent, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolParams

from openreward import AsyncOpenReward, OpenReward
from openreward.api.environments.types import (
    ImageBlock,
    Task as ORTask,
    TextBlock,
    ToolSpec,
)


class _RewardTracker:
    """Accumulates reward/finished signals across tool calls in a session."""

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
    session: Any,
    tracker: _RewardTracker,
) -> Tool:
    """Create an Inspect Tool backed by an async OR session."""
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
    """Convert OR prompt blocks to Inspect message content."""
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
    return content or ""


def _or_task_to_sample(or_task: ORTask, idx: int) -> Sample:
    """Convert an OpenReward Task to an Inspect Sample.

    The real prompt is fetched lazily when the session opens, so the
    Sample input here is a placeholder for display purposes.
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
    """Fetch OR tasks synchronously and convert to Inspect Samples."""
    with OpenReward() as client:
        env = client.environments.get(name=environment_name)
        or_tasks = env.list_tasks(split=split)

    if n_tasks is not None:
        or_tasks = or_tasks[:n_tasks]

    return [_or_task_to_sample(t, idx) for idx, t in enumerate(or_tasks)]


@scorer(metrics=[mean(), stderr()])
def _openreward_scorer() -> Scorer:
    """Reads the reward signal stored by openreward_solver."""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        reward = float(state.metadata.get("or_reward", 0.0))
        finished = bool(state.metadata.get("or_finished", False))
        return Score(
            value=reward,
            answer="FINISHED" if finished else "UNFINISHED",
            explanation=f"reward={reward}, finished={finished}",
        )

    return score


@solver
def openreward_solver(
    agent: Callable[[list[Tool]], Agent] | None = None,
) -> Solver:
    """Solver that manages an OpenReward session and runs an agent.

    Opens an async OR session per sample, fetches the real prompt,
    converts OR tools into Inspect tools with reward tracking, and
    delegates to the agent.

    Args:
        agent: Factory receiving OR-backed tools and returning an Agent.
            Defaults to ``react(tools=tools)``.
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
            prompt_blocks = await session.get_prompt()
            state.messages = [ChatMessageUser(content=_blocks_to_input(prompt_blocks))]

            tool_specs = await session.list_tools()
            tools = [_make_async_tool(spec, session, tracker) for spec in tool_specs]

            state = await as_solver(agent_factory(tools))(state, generate)

        state.metadata["or_reward"] = tracker.reward
        state.metadata["or_finished"] = tracker.finished
        return state

    return solve


@task
def openreward(
    environment: str,
    split: str = "train",
    n_tasks: int | None = None,
    agent: Callable[[list[Tool]], Agent] | None = None,
    **task_kwargs: Any,
) -> Task:
    """Inspect Task backed by an OpenReward environment.

    Args:
        environment: OpenReward environment name,
            e.g. ``"kanishk/EndlessTerminals"``.
        split: Dataset split (default ``"train"``).
        n_tasks: Cap on number of tasks. ``None`` for all.
        agent: Factory receiving OR-backed tools and returning an Agent.
            Defaults to ``react(tools=tools)``.
        **task_kwargs: Passed through to ``Task()`` — e.g.
            ``message_limit``, ``epochs``, ``sandbox``.
    """
    samples = _load_samples(environment, split, n_tasks)
    return Task(
        dataset=samples,
        solver=openreward_solver(agent=agent),
        scorer=_openreward_scorer(),
        **task_kwargs,
    )
