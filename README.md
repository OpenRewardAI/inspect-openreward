# inspect-openreward

Bridge package converting between [Inspect AI](https://inspect.aisi.org.uk/)
tools and the [OpenReward](https://docs.openreward.ai/) specification.

## Install

```bash
uv venv
uv sync
```

## Usage

```python
from inspect_openreward import openreward_tool_to_inspect

inspect_tool = openreward_tool_to_inspect(tool_spec, session)
```
