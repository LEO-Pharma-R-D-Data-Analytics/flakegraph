"""Background lease heartbeats for long-running claimed work.

Snowflake leases make crash recovery possible, but long OCR/LLM batches need a
separate heartbeat so another worker does not reclaim an actively processed
batch just because it exceeded the original lease window.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class LeaseHeartbeat:
    """Context manager that heartbeats a leased job on a daemon thread."""

    def __init__(
        self,
        heartbeat: Callable[[], None],
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.heartbeat = heartbeat
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    def __enter__(self) -> LeaseHeartbeat:
        self._thread = threading.Thread(
            target=self._run,
            name="flakegraph-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds)
        if exc_type is None and self.last_error is not None:
            raise self.last_error

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.heartbeat()
            except Exception as exc:  # pragma: no cover - thread timing guard
                self.last_error = exc
                self._stop.set()


def heartbeat_interval_seconds(lease_seconds: int) -> int:
    """Choose a heartbeat interval that is frequent but bounded."""

    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    # Heartbeat at one third of the lease, capped to a minute so long leases
    # still produce useful liveness signals, and floored at one second for tests
    # and intentionally short leases.
    return max(1, min(60, lease_seconds // 3))
