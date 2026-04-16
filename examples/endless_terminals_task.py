"""Evaluate an agent on OpenReward EndlessTerminals using the task factory.

This example uses the high-level ``openreward()`` task factory, which
handles session lifecycle, tool wiring, and reward scoring automatically.

Requires OPENAI_API_KEY and OPENREWARD_API_KEY to be set (e.g. via .env).
"""

from __future__ import annotations

from dotenv import load_dotenv
from inspect_ai import eval

from inspect_openreward import openreward

MODEL_NAME = "gpt-4o"


def main() -> None:
    load_dotenv()

    eval(
        openreward(
            environment="kanishk/EndlessTerminals",
            split="train",
            n_tasks=1,
        ),
        model=f"openai/{MODEL_NAME}",
    )


if __name__ == "__main__":
    main()
