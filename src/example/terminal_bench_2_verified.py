"""Run an Inspect eval against an OpenReward environment.

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
def terminal_bench_2_verified() -> Task:
    env = OpenReward().environments.get(
        name="GeneralReasoning/terminal-bench-2-verified" # https://openreward.ai/GeneralReasoning/terminal-bench-2-verified
    )
    return Task(
        dataset=openreward_dataset(env, split="test", limit=1), # Take one sample from the test split
        solver=openreward_solver(env, toolset="claude-code"), # Use the environment with the Claude Code toolset enabled
        scorer=openreward_scorer(),
    )


@task
def terminal_bench_2_verified_with_custom_system_prompt() -> Task:
    """Same environment, but with a custom solver chain plugged into the wrapper."""
    env = OpenReward().environments.get(
        name="GeneralReasoning/terminal-bench-2-verified"
    )
    return Task(
        dataset=openreward_dataset(env, split="test", limit=1),
        solver=openreward_solver(
            env,
            chain(
                system_message( # Add a custom system message
                    "Think carefully about each step before calling a tool."
                ),
                generate(tool_calls="loop"), # Generate output until state.completed=True
            ),
            toolset="claude-code"
        ),
        scorer=openreward_scorer(),
    )


def main() -> None:
    load_dotenv()
    eval(terminal_bench_2_verified, model=f"openai/{MODEL_NAME}")
    eval(
        terminal_bench_2_verified_with_custom_system_prompt,
        model=f"openai/{MODEL_NAME}",
    )


if __name__ == "__main__":
    main()
