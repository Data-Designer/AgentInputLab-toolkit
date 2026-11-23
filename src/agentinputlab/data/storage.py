"""Read-only destination-space checks for bounded data preparation.

These checks are snapshots, not filesystem reservations. Independent writers can
consume space later; callers must still bound each write and clean partial data.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from agentinputlab.contracts import ArtifactError, ValidationError

DEFAULT_PREPARED_LIMIT = 512 * 1024 * 1024
DEFAULT_DISK_RESERVE = 1024**3


def positive_byte_limit(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive integer")
    return value


def require_storage_capacity(
    allocations: Iterable[tuple[str | Path, int]],
    *,
    reserve_bytes: int = DEFAULT_DISK_RESERVE,
) -> None:
    """Check all planned bytes, summing paths on the same destination device.

    No directory or temporary file is created. A missing destination is checked
    using its nearest existing ancestor, not the current working directory.
    ``required`` is an upper bound on *additional* source/prepared bytes; cached
    sources do not need a second allowance. Unknown or negative sizes are errors.
    """
    positive_byte_limit(reserve_bytes, name="reserve_bytes")
    devices: dict[int, tuple[Path, int]] = {}
    for destination, required in allocations:
        if isinstance(required, bool) or not isinstance(required, int) or required < 0:
            raise ValidationError("planned storage bytes must be a non-negative integer")
        ancestor = Path(destination).expanduser().resolve()
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise ArtifactError(f"storage destination ancestor is not a directory: {ancestor}")
        device = ancestor.stat().st_dev
        previous = devices.get(device, (ancestor, 0))
        devices[device] = (previous[0], previous[1] + required)
    for ancestor, required in devices.values():
        free = shutil.disk_usage(ancestor).free
        if free < required + reserve_bytes:
            raise ArtifactError(
                f"insufficient storage at {ancestor}: need {required:,} additional bytes "
                f"plus {reserve_bytes:,} reserve, found {free:,} free; "
                "defer this source or use a source-side bounded sample"
            )
