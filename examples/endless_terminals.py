"""Run an Inspect AI react agent against an OpenReward environment.

Loosely based on the OpenReward quickstart. Assumes OPENAI_API_KEY and
OPENREWARD_API_KEY are set (e.g. via a .env file).
"""

from __future__ import annotations

from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from openreward import OpenReward

from inspect_openreward import openreward_tool_to_inspect

MODEL_NAME = "gpt-5.4"


def main() -> None:
    load_dotenv()

    or_client = OpenReward()
    environment = or_client.environments.get(name="kanishk/EndlessTerminals")
    tasks = environment.list_tasks(split="train")
    example_task = tasks[0]

    with environment.session(task=example_task) as session:
        or_client.rollout.create(
            run_name="EndlessTerminals-train-quickstart",
            rollout_name="example_task",
            environment="kanishk/EndlessTerminals",
            split="train",
            print_messages=True,
        )

        prompt = session.get_prompt()
        inspect_tools = [
            openreward_tool_to_inspect(t, session) for t in session.list_tools()
        ]

        agent = react(tools=inspect_tools, attempts=1)

        @task
        def openreward_example() -> Task:
            return Task(
                dataset=[Sample(input=prompt[0].text)],
                solver=agent,
            )

        eval(openreward_example, model=f"openai/{MODEL_NAME}")
        or_client.rollout.close()


if __name__ == "__main__":
    main()
