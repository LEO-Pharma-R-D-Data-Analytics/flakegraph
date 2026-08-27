"""Hold, order, and admit document-parsing work in front of a MinerU pool.

MinerU answers **409 when it is busy** and FlakeGraph's HTTP transport does not
retry, so without something in between a saturated pool does not slow down — it
drops documents. The shim exists to hold that work instead.

Holding is also the only way to enforce priority on this plane. There is no
scheduler inside MinerU to hand a band to, so ordering has to happen before
dispatch: a waiting request goes into PostgreSQL and is admitted only when the
pool has room. PostgreSQL rather than process memory because ordering has to hold
*across* shim replicas — two replicas each ordering their own callers correctly
still serves them in the wrong order relative to each other — and because a
restart should not lose work that a client is still waiting on.

Bands follow the same convention as the inference plane: **lower is served
first**. The pipeline's own task queue orders the other way; these are different
queues and the shim shares its vocabulary with the sidecar, not with the planner.

Tracking how much of the pool is busy is what admission control requires anyway,
so dispatching to the least-loaded replica costs nothing extra.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from kg_processor.serving.priority import ConsumerKeyring, load_keyring

logger = logging.getLogger(__name__)

# Every admission decision is made under this lock, so counting the busy pool and
# claiming a slot cannot interleave between replicas and overshoot capacity.
ADMISSION_LOCK_KEY = 0x0CD5_11
# ``/health`` has its own route, declared before the catch-all, so it never
# reaches this set.
UNAUTHENTICATED_PATHS = frozenset({"/ping", "/metrics"})


class OcrShimConfig(BaseModel):
    """Configure the shim from the environment the pod provides."""

    database_url: str
    upstream_host: str
    upstream_port: int = Field(default=8080, gt=0, lt=65536)
    # How many requests one parsing replica accepts before it starts answering
    # 409. The shim never admits more than this times the replicas it can see.
    upstream_capacity: int = Field(default=8, gt=0)
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8080, gt=0, lt=65536)
    keys_file: Path = Path("/etc/flakegraph/ocr/ocr-keys.json")
    bands: dict[str, int] | None = None
    poll_interval_seconds: float = Field(default=0.5, gt=0)
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    # A waiting or dispatched row whose owner stopped renewing it is reclaimed,
    # so a shim that dies mid-request does not permanently consume a slot.
    stale_after_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OcrShimConfig:
        """Build a configuration from ``FLAKEGRAPH_OCR_SHIM_*`` variables."""

        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field, variable in (
            ("database_url", "FLAKEGRAPH_OCR_SHIM_DATABASE_URL"),
            ("upstream_host", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"),
            ("upstream_port", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_PORT"),
            ("upstream_capacity", "FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY"),
            ("listen_host", "FLAKEGRAPH_OCR_SHIM_LISTEN_HOST"),
            ("listen_port", "FLAKEGRAPH_OCR_SHIM_LISTEN_PORT"),
            ("keys_file", "FLAKEGRAPH_OCR_SHIM_KEYS_FILE"),
            ("poll_interval_seconds", "FLAKEGRAPH_OCR_SHIM_POLL_INTERVAL_SECONDS"),
            ("request_timeout_seconds", "FLAKEGRAPH_OCR_SHIM_REQUEST_TIMEOUT_SECONDS"),
            ("stale_after_seconds", "FLAKEGRAPH_OCR_SHIM_STALE_AFTER_SECONDS"),
        ):
            if variable in env:
                values[field] = env[variable]
        if "FLAKEGRAPH_OCR_SHIM_BANDS" in env:
            values["bands"] = json.loads(env["FLAKEGRAPH_OCR_SHIM_BANDS"])
        return cls.model_validate(values)


class UpstreamPool:
    """Resolve parsing replicas and hand out the least-loaded one.

    The shim is configured with one name, not a list of addresses, so the pool
    can autoscale underneath it without a configuration change. Resolution is a
    DNS lookup of that name, which is how any client-side balancer finds its
    backends — the shim still learns nothing about the cluster it runs in.
    """

    def __init__(
        self,
        host: str,
        port: int,
        capacity_per_replica: int,
        resolver: Callable[[], Awaitable[tuple[str, ...]]] | None = None,
    ) -> None:
        """Record the pool's name and the load each replica will accept."""

        self._host = host
        self._port = port
        self._capacity_per_replica = capacity_per_replica
        self._resolver = resolver
        self._in_flight: dict[str, int] = {}
        self._endpoints: tuple[str, ...] = ()

    async def _resolve(self) -> tuple[str, ...]:
        """Return the pool's current addresses, or what was last seen on failure."""

        if self._resolver is not None:
            return await self._resolver()
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(self._host, self._port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            logger.warning("could not resolve parsing pool %s", self._host)
            return self._endpoints
        return tuple(sorted({str(info[4][0]) for info in infos}))

    async def refresh(self) -> tuple[str, ...]:
        """Re-resolve the pool, forgetting counters for replicas that are gone."""

        resolved = await self._resolve()
        if resolved:
            self._endpoints = resolved
            self._in_flight = {
                address: self._in_flight.get(address, 0) for address in self._endpoints
            }
        return self._endpoints

    @property
    def capacity(self) -> int:
        """Return how many requests the whole pool can hold right now."""

        return len(self._endpoints) * self._capacity_per_replica

    def acquire(self) -> str | None:
        """Claim the least-loaded replica, or ``None`` when every one is full."""

        candidates = [
            (count, address)
            for address, count in self._in_flight.items()
            if count < self._capacity_per_replica
        ]
        if not candidates:
            return None
        _, address = min(candidates)
        self._in_flight[address] += 1
        return address

    def release(self, address: str) -> None:
        """Return a slot after a request completes, however it completed."""

        if address in self._in_flight:
            self._in_flight[address] = max(0, self._in_flight[address] - 1)

    def base_url(self, address: str) -> str:
        """Return the URL for a resolved replica address."""

        formatted = f"[{address}]" if ":" in address else address
        return f"http://{formatted}:{self._port}"


class OcrQueue:
    """Order waiting requests across every shim replica and admit them in turn."""

    def __init__(self, pool: Any, owner: str, stale_after_seconds: float) -> None:
        """Hold the connection pool and the identity this replica renews under."""

        self._pool = pool
        self._owner = owner
        self._stale_after_seconds = stale_after_seconds

    async def enqueue(self, request_id: str, priority: int, consumer_class: str) -> None:
        """Record a request as waiting, in priority then arrival order."""

        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO flakegraph_ocr_request
                    (id, priority, consumer_class, status, shim_owner)
                VALUES (%s, %s, %s, 'waiting', %s)
                """,
                (request_id, priority, consumer_class, self._owner),
            )

    async def try_admit(self, request_id: str, capacity: int) -> bool:
        """Claim a slot when the pool has room and nothing better is waiting.

        Counting the busy pool and claiming a slot happen inside one transaction
        holding the advisory lock. Without that, two replicas could both read the
        same free-slot count and both dispatch into it.
        """

        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(%s)", (ADMISSION_LOCK_KEY,))
            await connection.execute(
                """
                DELETE FROM flakegraph_ocr_request
                WHERE heartbeat_at < CURRENT_TIMESTAMP - make_interval(secs => %s)
                """,
                (self._stale_after_seconds,),
            )
            cursor = await connection.execute(
                "SELECT count(*) AS busy FROM flakegraph_ocr_request WHERE status = 'dispatched'"
            )
            row = await cursor.fetchone()
            free = capacity - int(row["busy"])
            if free <= 0:
                return False
            cursor = await connection.execute(
                """
                SELECT id FROM flakegraph_ocr_request
                WHERE status = 'waiting'
                ORDER BY priority, created_at, id
                LIMIT %s
                """,
                (free,),
            )
            admissible = {record["id"] for record in await cursor.fetchall()}
            if request_id not in admissible:
                return False
            await connection.execute(
                """
                UPDATE flakegraph_ocr_request
                SET status = 'dispatched', heartbeat_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (request_id,),
            )
            return True

    async def renew(self, request_id: str) -> None:
        """Keep a row alive so it is not reclaimed while a client still waits."""

        async with self._pool.connection() as connection:
            await connection.execute(
                "UPDATE flakegraph_ocr_request SET heartbeat_at = CURRENT_TIMESTAMP WHERE id = %s",
                (request_id,),
            )

    async def release(self, request_id: str) -> None:
        """Drop a finished request, freeing its slot for whatever waits next."""

        async with self._pool.connection() as connection:
            await connection.execute(
                "DELETE FROM flakegraph_ocr_request WHERE id = %s", (request_id,)
            )


def create_app(
    config: OcrShimConfig,
    keyring: ConsumerKeyring | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    queue: OcrQueue | None = None,
    upstreams: UpstreamPool | None = None,
) -> FastAPI:
    """Build the shim application in front of one parsing pool.

    The queue and the pool are the two systems outside this process, so both are
    injectable at the composition root rather than reached for internally.
    """

    resolved = keyring if keyring is not None else load_keyring(config.keys_file, config.bands)
    pool_view = upstreams or UpstreamPool(
        config.upstream_host, config.upstream_port, config.upstream_capacity
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open the database pool and one upstream client for the process."""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
        ) as client:
            app.state.client = client
            if queue is not None:
                app.state.queue = queue
                await pool_view.refresh()
                yield
                return
            async with AsyncConnectionPool(
                config.database_url,
                min_size=1,
                max_size=8,
                open=False,
                kwargs={"row_factory": dict_row, "autocommit": True},
            ) as connections:
                app.state.queue = OcrQueue(connections, _owner_id(), config.stale_after_seconds)
                await pool_view.refresh()
                yield

    app = FastAPI(
        title="FlakeGraph OCR shim",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.keyring = resolved
    app.state.upstreams = pool_view

    @app.get("/health")
    async def health() -> JSONResponse:
        """Report readiness without touching the parsing pool."""

        return JSONResponse({"status": "ok", "replicas": len(await pool_view.refresh())})

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def parse(path: str, request: Request) -> Response:
        """Authenticate, hold the request until the pool has room, then forward."""

        route = f"/{path}"
        if route in UNAUTHENTICATED_PATHS:
            return JSONResponse({"status": "ok"})

        consumer_class = resolved.classify(_presented_key(request))
        if consumer_class is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        priority = resolved.priority_for(consumer_class)
        request_id = str(uuid4())
        body = await request.body()
        held: OcrQueue = request.app.state.queue

        await held.enqueue(request_id, priority, consumer_class)
        address: str | None = None
        try:
            address = await _await_slot(
                held, pool_view, request_id, priority, consumer_class, config
            )
            return await _forward(request, route, body, address, pool_view, held, request_id)
        except Exception:
            if address is not None:
                pool_view.release(address)
            await held.release(request_id)
            raise

    return app


def run(config: OcrShimConfig | None = None) -> None:
    """Serve the shim until the process is signalled."""

    resolved = config if config is not None else OcrShimConfig.from_env()
    uvicorn.run(
        create_app(resolved),
        host=resolved.listen_host,
        port=resolved.listen_port,
        log_level="info",
    )


async def _await_slot(
    queue: OcrQueue,
    upstreams: UpstreamPool,
    request_id: str,
    priority: int,
    consumer_class: str,
    config: OcrShimConfig,
) -> str:
    """Wait until this request is both next in line and has a replica to go to.

    The client is held here rather than refused. That is the whole point: a 409
    reaching a caller with no retry loses the document.
    """

    while True:
        await upstreams.refresh()
        if upstreams.capacity and await queue.try_admit(request_id, upstreams.capacity):
            address = upstreams.acquire()
            if address is not None:
                return address
            # The pool shrank between the admission decision and the claim.
            # Return the slot and queue again at the band this caller actually
            # holds, so a lost race cannot promote or demote the request.
            await queue.release(request_id)
            await queue.enqueue(request_id, priority, consumer_class)
        await queue.renew(request_id)
        await asyncio.sleep(config.poll_interval_seconds)


async def _forward(
    request: Request,
    route: str,
    body: bytes,
    address: str,
    upstreams: UpstreamPool,
    queue: OcrQueue,
    request_id: str,
) -> Response:
    """Relay the held request to the chosen replica and free its slot after."""

    client: httpx.AsyncClient = request.app.state.client
    upstream = client.build_request(
        request.method,
        f"{upstreams.base_url(address)}{route}",
        params=request.query_params,
        headers=_forwarded_headers(request.headers),
        content=body,
    )
    response = await client.send(upstream, stream=True)

    async def _finish() -> None:
        await response.aclose()
        upstreams.release(address)
        await queue.release(request_id)

    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=dict(response.headers),
        background=BackgroundTask(_finish),
    )


def _owner_id() -> str:
    """Identify this replica so its rows can be reclaimed if the process dies."""

    return os.environ.get("POD_NAME") or socket.gethostname()


def _presented_key(request: Request) -> str:
    """Extract a bearer credential without treating a malformed header as valid."""

    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return credential.strip()


def _forwarded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop the caller's credential and anything scoped to this connection.

    The parsing pool sits behind the shim and has no authentication of its own,
    so relaying the key would only copy it into another process's logs.
    """

    dropped = {"host", "content-length", "authorization", "connection", "transfer-encoding"}
    return {name: value for name, value in headers.items() if name.lower() not in dropped}
