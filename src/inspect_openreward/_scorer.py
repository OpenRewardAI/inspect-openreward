from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from ._constants import FINISHED_METADATA_KEY, REWARD_METADATA_KEY


@scorer(metrics=[mean(), stderr()])
def openreward_scorer() -> Scorer:
    """Scorer reading the terminal OpenReward reward captured by `openreward_solver`.

    OpenReward environments return `reward` (and `finished=True`) inline on the
    final "submit"-style `ToolOutput`. The solver stashes the latest non-null
    reward into `state.metadata`; this scorer reads it back and returns a
    numeric `Score`.

    Samples that never produced a reward (e.g. ran out of turns before
    submitting) are scored `0.0` with `metadata={"finished": False}`.
    """

    async def score(state: TaskState, target: Target) -> Score:
        reward = state.metadata.get(REWARD_METADATA_KEY)
        finished = bool(state.metadata.get(FINISHED_METADATA_KEY, False))
        if reward is None:
            return Score(value=0.0, metadata={"finished": finished, "reward": None})
        return Score(
            value=float(reward),
            metadata={"finished": finished, "reward": reward},
        )

    return score
