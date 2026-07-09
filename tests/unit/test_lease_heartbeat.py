from __future__ import annotations

import pytest

from kg_processor.application.lease_heartbeat import heartbeat_interval_seconds


def test_heartbeat_interval_uses_one_third_of_lease_with_bounds() -> None:
    assert heartbeat_interval_seconds(3) == 1
    assert heartbeat_interval_seconds(90) == 30
    assert heartbeat_interval_seconds(900) == 60


def test_heartbeat_interval_rejects_non_positive_lease() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        heartbeat_interval_seconds(0)
