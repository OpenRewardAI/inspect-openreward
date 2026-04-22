from __future__ import annotations

from typing import Optional

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from openreward.api.environments.client import Environment

from ._constants import TASK_METADATA_KEY


def openreward_dataset(
    environment: Environment,
    split: str,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Dataset:
    """Build an Inspect `Dataset` from an OpenReward environment split.

    Each `Sample` carries the OpenReward `Task` in `metadata` under the
    `"openreward_task"` key. `Sample.input` is left empty — `openreward_solver`
    fills it from `session.get_prompt()` at solve time, which is where the
    authoritative prompt (including images) lives.

    Args:
        environment: An OpenReward `Environment` (from `client.environments.get(...)`).
        split: Split name, e.g. `"train"` or `"test"`.
        limit: If set, only the first `limit` tasks are loaded.
        shuffle: Shuffle the samples after loading.
        seed: Seed for shuffling.

    Returns:
        An Inspect `MemoryDataset`.
    """
    tasks = environment.list_tasks(split=split)
    if limit is not None:
        tasks = tasks[:limit]

    samples = [
        Sample(
            id=f"{split}-{i}",
            input="",
            metadata={TASK_METADATA_KEY: task, "openreward_split": split},
        )
        for i, task in enumerate(tasks)
    ]

    dataset = MemoryDataset(
        samples=samples,
        name=f"openreward:{environment.deployment_name}:{split}",
    )
    if shuffle:
        dataset.shuffle(seed=seed)
    return dataset
