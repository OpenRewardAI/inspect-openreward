"""Run an Inspect AI eval against an OpenReward environment.

Assumes OPENAI_API_KEY and OPENREWARD_API_KEY are set (e.g. via a .env file).
"""

from __future__ import annotations

from dotenv import load_dotenv
from inspect_ai import Task, eval, task
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


def main() -> None:
    load_dotenv()
    eval(endless_terminals, model=f"openai/{MODEL_NAME}")
    print("")


if __name__ == "__main__":
    main()
