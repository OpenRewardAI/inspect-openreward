"""Run an Inspect AI eval against an OpenReward environment.

Assumes OPENAI_API_KEY and OPENREWARD_API_KEY are set (e.g. via a .env file).
"""

from __future__ import annotations

from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.solver import chain, generate, system_message
from openreward import OpenReward

from inspect_openreward import (
    openreward_dataset,
    openreward_scorer,
    openreward_solver,
)

MODEL_NAME = "gpt-5.4"


@task
def endless_terminals() -> Task:
    env = OpenReward().environments.get(name="kanishk/EndlessTerminals")
    return Task(
        dataset=openreward_dataset(env, split="train", limit=1),
        solver=openreward_solver(env),
        scorer=openreward_scorer(),
    )


@task
def endless_terminals_with_custom_system_prompt() -> Task:
    """Same environment, but with a custom solver chain plugged into the wrapper."""
    env = OpenReward().environments.get(name="kanishk/EndlessTerminals")
    return Task(
        dataset=openreward_dataset(env, split="train", limit=1),
        solver=openreward_solver(
            env,
            chain(
                system_message(
                    "Think carefully about each step before calling a tool."
                ),
                generate(tool_calls="loop"),
            ),
        ),
        scorer=openreward_scorer(),
    )


def main() -> None:
    load_dotenv()
    eval(endless_terminals, model=f"openai/{MODEL_NAME}")
    eval(endless_terminals_with_custom_system_prompt, model=f"openai/{MODEL_NAME}")


if __name__ == "__main__":
    main()
