# inspect-openreward

Run [OpenReward](https://docs.openreward.ai/) environments as
[Inspect AI](https://inspect.aisi.org.uk/) evals.

Provides an Inspect-native `Dataset`, `Solver`, and `Scorer` for any OpenReward
environment, so you get Inspect's eval harness, transcript viewer, metrics,
and model abstraction for free — and OpenReward's tools, tasks, and rewards
surface as first-class Inspect primitives.

## Install

```bash
uv venv
uv sync
```

Set `OPENREWARD_API_KEY` and whichever model provider keys you need
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## Quickstart

```python
from inspect_ai import Task, eval, task
from openreward import OpenReward

from inspect_openreward import (
    openreward_dataset,
    openreward_scorer,
    openreward_solver,
)


@task
def endless_terminals() -> Task:
    env = OpenReward().environments.get(name="kanishk/EndlessTerminals")
    return Task(
        dataset=openreward_dataset(env, split="train", limit=10),
        solver=openreward_solver(env),
        scorer=openreward_scorer(),
    )


if __name__ == "__main__":
    eval(endless_terminals, model="openai/gpt-5.4")
```

See [`./src/example/endless_terminals.py`](examples/endless_terminals.py) for a runnable version.

## What each piece does

### `openreward_dataset(environment, split, limit=None, shuffle=False, seed=None)`

Builds an Inspect `Dataset` from an OpenReward environment split. Each `Sample`
carries the underlying OpenReward `Task` in metadata; the prompt is resolved
lazily by the solver (so image prompts work, and there's no network round-trip
at dataset-construction time).

### `openreward_solver(environment, toolset=None, tool_choice="auto")`

Inspect `@solver` that, per sample:

1. Opens `environment.session(task=..., toolset=toolset)`.
2. Fetches `session.get_prompt()` and injects it as the user message (text and
   image blocks both supported).
3. Converts `session.list_tools()` into Inspect tools, auto-detecting the
   provider from `state.model.api` and sanitising the JSON schema via
   `openreward.sanitize_tool_schema`.
4. Runs the react-style tool-call loop via Inspect's `generate(tool_calls="loop")`.
5. Captures the terminal `reward` / `finished` from tool outputs into
   `state.metadata` for the scorer to read.
6. Closes the session on teardown.

Pass a `toolset` (e.g. `"claude-code"`) to use an OpenReward harness-native
tool surface instead of the environment's own tools.

### `openreward_scorer()`

`@scorer` that reads the terminal reward captured by `openreward_solver` and
returns an Inspect `Score`. Metrics: `mean()` and `stderr()`. Samples that
never produced a reward (e.g. ran out of turns) score `0.0`.

### `openreward_tool_to_inspect(tool_spec, session, provider=None)`

Low-level helper for converting a single OpenReward `ToolSpec` into an Inspect
`Tool`. The `openreward_solver` is built on top of this; use it directly if
you want to compose tools differently.

## Model provider mapping

The solver maps Inspect's model-provider identifier to OpenReward's `Provider`
enum for schema sanitisation:

| Inspect `state.model.api` | OpenReward provider |
|---|---|
| `openai`, `openai-api` | `openai` |
| `anthropic` | `anthropic` |
| `google`, `vertex` | `google` |
| `openrouter` | `openrouter` |
| anything else | `openai` (safest superset) |
