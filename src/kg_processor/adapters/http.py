"""Reusable synchronous HTTP transport for provider adapters."""

from __future__ import annotations

from threading import Lock

import httpx


class HttpClientPool:
    """Reuse thread-safe HTTP clients while preserving per-request timeouts.

    Graph extraction can issue thousands of independent provider calls from a
    bounded thread pool. Constructing a client for every call discards keep-alive
    connections and repeatedly pays DNS, TCP, and TLS setup costs. HTTPX clients
    are safe to share between threads, so this pool retains one client for each
    configured timeout used by a provider instance.

    Timeout-specific clients preserve the existing adapter contract: a structured
    request may override the provider's default timeout. Creation is protected by
    a small lock, while requests themselves run concurrently through HTTPX's own
    connection pool. Clients are released when the adapter is closed or collected.
    """

    def __init__(self) -> None:
        """Initialize an empty pool without opening any network connection."""

        self._clients: dict[float, httpx.Client] = {}
        self._lock = Lock()

    def client(self, timeout_seconds: float) -> httpx.Client:
        """Return the shared client configured for ``timeout_seconds``.

        HTTPX opens sockets lazily, so creating the object under the lock does not
        serialize network activity. The same object is returned to every caller
        using the same timeout, enabling connection reuse under parallel load.
        """

        if timeout_seconds <= 0:
            raise ValueError("HTTP client timeout must be positive")
        with self._lock:
            client = self._clients.get(timeout_seconds)
            if client is None:
                client = httpx.Client(timeout=timeout_seconds)
                self._clients[timeout_seconds] = client
            return client

    def close(self) -> None:
        """Close all retained connection pools and make repeated closure harmless."""

        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()
