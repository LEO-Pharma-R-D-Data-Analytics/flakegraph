"""Contracts for provider HTTP connection reuse and deterministic cleanup."""

from __future__ import annotations

from typing import Any

import pytest

from kg_processor.adapters.http import HttpClientPool


class _FakeClient:
    """Record one timeout-specific client without opening network connections."""

    def __init__(self, *, timeout: float) -> None:
        """Retain the requested timeout and initialize close accounting."""

        self.timeout = timeout
        self.close_calls = 0

    def close(self) -> None:
        """Record deterministic connection-pool release."""

        self.close_calls += 1


def test_http_client_pool_reuses_timeout_specific_clients(monkeypatch: Any) -> None:
    """Equal timeouts share sockets while distinct policies remain isolated."""

    created: list[_FakeClient] = []

    def build_client(*, timeout: float) -> _FakeClient:
        """Return a recorded fake for HTTPX's keyword-only construction shape."""

        client = _FakeClient(timeout=timeout)
        created.append(client)
        return client

    monkeypatch.setattr("kg_processor.adapters.http.httpx.Client", build_client)
    pool = HttpClientPool()

    first = pool.client(30.0)
    assert pool.client(30.0) is first
    second = pool.client(90.0)

    assert second is not first
    assert [client.timeout for client in created] == [30.0, 90.0]
    pool.close()
    pool.close()
    assert [client.close_calls for client in created] == [1, 1]


def test_http_client_pool_rejects_non_positive_timeout() -> None:
    """Invalid timeout policies fail before allocating a transport."""

    with pytest.raises(ValueError, match="timeout must be positive"):
        HttpClientPool().client(0)
