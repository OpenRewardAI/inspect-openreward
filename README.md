# inspect-openreward

[![PyPI version](https://img.shields.io/pypi/v/inspect-openreward)](https://pypi.org/project/inspect-openreward/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-green)](https://pypi.org/project/inspect-openreward/)

Run [OpenReward](https://docs.openreward.ai/) environments as [Inspect](https://inspect.aisi.org.uk/) evals.

Provides an Inspect-native `Dataset`, `Scorer`, and a session-lifecycle wrapper solver for any OpenReward environment, so you get Inspect's eval harness, transcript viewer, metrics, and model abstraction for free — and OpenReward's tools, tasks, and rewards surface as first-class Inspect primitives. The wrapper takes an arbitrary inner solver chain, so you can keep the default react-style loop or plug in your own scaffolding (`system_message`, `basic_agent`, `react`, custom `@solver`, …) without touching session management.

## Install

```bash
uv venv
uv sync
```

Set `OPENREWARD_API_KEY` and whichever model provider keys you need (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## Quickstart

See [`./src/example/terminal_bench_2_verified.py`](./src/example/terminal_bench_2_verified.py) for a runnable version with both a default, and a customer solver chain, task.

## Custom solver chains

`openreward_solver` is a wrapper: it owns session open/close, prompt injection, tool conversion + installation, and reward capture, and then runs whatever inner solver chain you hand it. That means any Inspect solver composition slots in — chain together `system_message`, `prompt_template`, `chain_of_thought`, `self_critique`, `use_tools(..., append=True)`, `basic_agent`, `react`, or your own `@solver` — and the OpenReward session-bound tools remain available throughout:

```python
from inspect_ai.solver import chain, generate, system_message

@task
def terminal_bench_2_verified_custom() -> Task:
    env = OpenReward().environments.get(name="GeneralReasoning/terminal-bench-2-verified")
    return Task(
        dataset=openreward_dataset(env, split="test", limit=10),
        solver=openreward_solver(
            env,
            chain(
                system_message("Think carefully before each tool call."),
                generate(tool_calls="loop"),
            ),
            toolset="claude-code"
        ),
        scorer=openreward_scorer(),
    )
```

Reward / `finished` capture is baked into the installed tools, so it keeps working no matter how the inner chain drives tool calls (`generate(...)`, `execute_tools(...)` inside `basic_agent`, a custom loop, …).

## What each piece does

### `openreward_dataset(environment, split, limit=None, shuffle=False, seed=None)`

Builds an Inspect `Dataset` from an OpenReward environment split. Each `Sample` carries the underlying OpenReward `Task` in metadata; the prompt is resolved lazily by the solver (so image prompts work, and there's no network round-trip at dataset-construction time).

### `openreward_solver(environment, solver=None, *, toolset=None, tool_choice="auto")`

Session-lifecycle wrapper around an arbitrary inner solver chain. Per sample it:

1. Opens `environment.session(task=..., toolset=toolset)`.
2. Fetches `session.get_prompt()` and injects it as the user message (text and image blocks both supported).
3. Converts `session.list_tools()` into Inspect tools, auto-detecting the provider from `state.model.api` and sanitising the JSON schema via `openreward.sanitize_tool_schema`, and installs them on `state.tools` / `state.tool_choice`.
4. Runs the inner `solver` inside the open session. If `solver=None` (the default), runs `generate(tool_calls="loop")` — the react-style loop. Pass a `Solver` or a `list[Solver]` (composed via `chain(...)`) to customise.
5. Captures the terminal `reward` / `finished` from tool outputs into `state.metadata` for the scorer to read — regardless of how the inner chain invokes tools.
6. Closes the session on teardown.

**Arguments**

- `environment`: the OpenReward `Environment` to open sessions against. The dataset should be built from the same environment.
- `solver`: inner solver (or `list[Solver]`, normalised via `inspect_ai.solver.chain`). Defaults to `generate(tool_calls="loop")`.
- `toolset` *(keyword-only)*: optional OpenReward toolset name passed to `environment.session(...)` — e.g. `"claude-code"` for a harness-native bash tool surface in addition to the environment's own tools.
- `tool_choice` *(keyword-only)*: passed through to Inspect's `state.tool_choice`.

Inner chains can layer on further tools via `use_tools(extra_tools, append=True)` — the session-bound tools installed by the wrapper remain available.

### `openreward_scorer()`

`@scorer` that reads the terminal reward captured by `openreward_solver` and returns an Inspect `Score`. Metrics: `mean()` and `stderr()`. Samples that never produced a reward (e.g. ran out of turns) score `0.0`.

### `openreward_tool_to_inspect(tool_spec, session, provider=None)`

Low-level helper for converting a single OpenReward `ToolSpec` into an Inspect `Tool`. The `openreward_solver` wrapper is built on top of this; use it directly if you want to bypass the wrapper and manage the session yourself.

## Model provider mapping

The solver maps Inspect's model-provider identifier to OpenReward's `Provider` enum for schema sanitisation:

| Inspect `state.model.api` | OpenReward provider |
|---|---|
| `openai`, `openai-api` | `openai` |
| `anthropic` | `anthropic` |
| `google`, `vertex` | `google` |
| `openrouter` | `openrouter` |
| anything else | `openai` (safest superset) |
