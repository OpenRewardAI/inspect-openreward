"""Evaluate an agent on OpenReward EndlessTerminals (low-level approach).

Manually manages the OR session and wires tools into an Inspect task.
For most users, the task factory in ``endless_terminals_task.py`` is simpler.

Requires OPENAI_API_KEY and OPENREWARD_API_KEY to be set (e.g. via .env).
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.tool import ToolDef, ToolParams
from openreward import OpenReward

MODEL_NAME = "gpt-5.4"


def main() -> None:
    load_dotenv()

    or_client = OpenReward()
    environment = or_client.environments.get(name="kanishk/EndlessTerminals")
    tasks = environment.list_tasks(split="train")
    example_task = tasks[0]

    with environment.session(task=example_task) as session:
        prompt = session.get_prompt()

        def _make_tool(spec: Any):
            name = spec.name

            async def execute(**kwargs: Any) -> str:
                result = session.call_tool(name, kwargs)
                return result.blocks[0].text if result.blocks else ""

            return ToolDef(
                tool=execute,
                name=name,
                description=spec.description,
                parameters=ToolParams(**(spec.input_schema or {})),
            ).as_tool()

        inspect_tools = [_make_tool(t) for t in session.list_tools()]
        agent = react(tools=inspect_tools, attempts=1)

        @task
        def openreward_example() -> Task:
            return Task(
                dataset=[Sample(input=prompt[0].text)],
                solver=agent,
            )

        eval(openreward_example, model=f"openai/{MODEL_NAME}")


if __name__ == "__main__":
    main()
