"""The enforcement floor in front of an inference engine.

The engine binds ``127.0.0.1`` and this process owns the only exposed port, so
there is no route to inference that skips authentication or priority stamping.
Everything here speaks the OpenAI wire format and nothing else, which is what
lets the engine behind it be replaced without touching consumers.

Two properties are deliberate and easy to lose:

* Client-supplied priority is **removed** from every JSON body, on every path,
  before anything else happens. Stamping only the generation paths would leave
  the others usable as a forgery channel.
* The server's value is stamped **unconditionally**. vLLM treats a missing
  ``priority`` as ``0``, which is its *highest* band, so a request that slips
  through unstamped is promoted rather than dropped.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from kg_processor import __version__
from kg_processor.serving.priority import ConsumerKeyring, load_keyring
from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    SizingVerdict,
    compute_sizing,
)

logger = logging.getLogger(__name__)

# Read-only and carrying no inference. The kubelet probes the health routes and
# the endpoint picker scores replicas on the metrics one, and both reach the pod
# before any consumer holds a key.
#
# The render routes are here for the same reason and with the same limit. The
# picker has to turn a prompt into the engine's own token ids to know which
# replica already holds that prefix - an estimate cannot match, because the
# comparison is against real token ids in the cache events. Rendering allocates
# no KV blocks, runs no forward pass and returns no completion: it is the
# tokenizer, reachable only from whatever the network policy admits to this
# port, which is the picker. Guarding it with a key would mean stamping a
# priority band onto a call that never reaches the scheduler.
UNAUTHENTICATED_PATHS = frozenset(
    {
        "/health",
        "/ping",
        "/metrics",
        "/v1/chat/completions/render",
        "/v1/completions/render",
    }
)

# Adapter management mutates what the pod serves. The platform pins one model per
# pod, so these are refused here rather than guarded by a key nobody should hold.
REFUSED_PATHS = frozenset({"/load_lora_adapter", "/unload_lora_adapter"})

# vLLM reads ``priority`` from the request body on its generation routes. Other
# routes have no use for it, and adding an unexpected field to their payloads
# risks a validation error from the engine.
STAMPED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/invocations",
    }
)

PRIORITY_FIELD = "priority"
PRIORITY_HEADERS = frozenset({"x-vllm-priority", "x-flakegraph-priority"})

# Metric labels have to stay a small fixed vocabulary: a label per key, request
# or free-form path would grow a series per value and eventually take the
# Prometheus server down with it. Paths collapse onto this set, everything else
# is ``other``, and a request that presented no usable key is one class.
ROUTE_LABELS: Mapping[str, str] = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
    "/v1/models": "models",
}
OTHER_ROUTE_LABEL = "other"
UNAUTHENTICATED_CLASS = "unauthenticated"
# A generation call can legitimately run for minutes, which the default buckets
# (topping out at ten seconds) would flatten into one bin.
REQUEST_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600)
# A scrape has its own deadline on the Prometheus side, and an engine that has
# stopped answering must not hold the sidecar's own series hostage to it.
SCRAPE_TIMEOUT_SECONDS = 5.0

# Headers that describe one connection and must not be relayed onto another.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


class SidecarConfig(BaseModel):
    """Configure the sidecar entirely from the environment the pod provides."""

    upstream_base_url: str = "http://127.0.0.1:8001"
    listen_host: str = "0.0.0.0"
    listen_port: int = Field(default=8000, gt=0, lt=65536)
    keys_file: Path = Path("/etc/flakegraph/serving-keys.json")
    bands: dict[str, int] | None = None
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    geometry: ModelGeometry | None = None
    budget: DeviceBudget | None = None
    expected_context_tokens: int = Field(default=32768, gt=0)
    max_num_seqs: int = Field(default=32, gt=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SidecarConfig:
        """Build a configuration from ``FLAKEGRAPH_SIDECAR_*`` variables.

        Sizing inputs are optional as a group: when the model geometry and device
        budget are both present the startup check runs, and when neither is
        present the sidecar serves without it. A partial group is an error, since
        silently skipping the check is the failure this design exists to prevent.
        """

        env = os.environ if environ is None else environ
        geometry = _optional_model(ModelGeometry, env, "FLAKEGRAPH_SIDECAR_GEOMETRY")
        budget = _optional_model(DeviceBudget, env, "FLAKEGRAPH_SIDECAR_DEVICE_BUDGET")
        if (geometry is None) != (budget is None):
            raise ValueError(
                "FLAKEGRAPH_SIDECAR_GEOMETRY and FLAKEGRAPH_SIDECAR_DEVICE_BUDGET "
                "must be set together or not at all"
            )
        values: dict[str, Any] = {"geometry": geometry, "budget": budget}
        for field, variable in (
            ("upstream_base_url", "FLAKEGRAPH_SIDECAR_UPSTREAM"),
            ("listen_host", "FLAKEGRAPH_SIDECAR_LISTEN_HOST"),
            ("listen_port", "FLAKEGRAPH_SIDECAR_LISTEN_PORT"),
            ("keys_file", "FLAKEGRAPH_SIDECAR_KEYS_FILE"),
            ("request_timeout_seconds", "FLAKEGRAPH_SIDECAR_REQUEST_TIMEOUT_SECONDS"),
            ("expected_context_tokens", "FLAKEGRAPH_SIDECAR_EXPECTED_CONTEXT_TOKENS"),
            ("max_num_seqs", "FLAKEGRAPH_SIDECAR_MAX_NUM_SEQS"),
        ):
            if variable in env:
                values[field] = env[variable]
        if "FLAKEGRAPH_SIDECAR_BANDS" in env:
            values["bands"] = json.loads(env["FLAKEGRAPH_SIDECAR_BANDS"])
        return cls.model_validate(values)

    def sizing_verdict(self) -> SizingVerdict | None:
        """Evaluate the configured sequence limit, or ``None`` when not declared."""

        if self.geometry is None or self.budget is None:
            return None
        return compute_sizing(
            self.geometry,
            self.budget,
            self.expected_context_tokens,
            self.max_num_seqs,
        )


class SidecarMetrics:
    """Own the series this process adds on top of the engine's.

    Each application gets its own registry rather than the process-wide default.
    The default one also carries the Python runtime collectors, and the engine
    already exports those under the same names - emitting them twice in one
    scrape body would make Prometheus reject the whole scrape.
    """

    def __init__(self) -> None:
        """Register every series up front so an idle sidecar still exposes them."""

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "flakegraph_sidecar_requests_total",
            "Requests answered by the sidecar, by the status the caller received.",
            ["consumer_class", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "flakegraph_sidecar_request_duration_seconds",
            "Time from receiving a request to sending its last byte.",
            ["consumer_class", "route"],
            buckets=REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "flakegraph_sidecar_requests_in_flight",
            "Requests received whose last byte has not yet been sent.",
            ["consumer_class"],
            registry=self.registry,
        )
        self.priority_stamped = Counter(
            "flakegraph_sidecar_priority_stamped_total",
            "Generation requests stamped with each priority band.",
            ["consumer_class", "priority"],
            registry=self.registry,
        )
        Info(
            "flakegraph_sidecar_build",
            "Version of the FlakeGraph package serving this sidecar.",
            registry=self.registry,
        ).info({"version": __version__})

    def render(self) -> bytes:
        """Serialise the sidecar's own series in the text exposition format."""

        return generate_latest(self.registry)


class _Accounting:
    """Close one request's metrics exactly once, when its last byte has left.

    On a streamed completion the headers leave long before the body, so the
    observation is attached to the stream rather than to the handler returning.
    """

    def __init__(self, metrics: SidecarMetrics, consumer_class: str, route: str) -> None:
        """Mark the request as in flight from the moment it was received."""

        self._metrics = metrics
        self._consumer_class = consumer_class
        self._route = ROUTE_LABELS.get(route, OTHER_ROUTE_LABEL)
        self._started = time.perf_counter()
        self._closed = False
        metrics.in_flight.labels(consumer_class).inc()

    def close(self, response: Response) -> Response:
        """Record a response that is complete the moment it is returned."""

        self._record(response.status_code)
        return response

    def abandon(self) -> None:
        """Record a request the handler gave up on before any response was built.

        Starlette answers an escaped exception with a 500, so that is the status
        the caller saw. Recording it here is what keeps the in-flight gauge from
        drifting upward by one for every engine timeout.
        """

        self._record(500)

    async def stream(self, chunks: AsyncIterator[bytes], status_code: int) -> AsyncIterator[bytes]:
        """Relay a body and record the request once the relay stops, however it stops."""

        try:
            async for chunk in chunks:
                yield chunk
        finally:
            self._record(status_code)

    def _record(self, status_code: int) -> None:
        if self._closed:
            return
        self._closed = True
        self._metrics.in_flight.labels(self._consumer_class).dec()
        self._metrics.requests.labels(self._consumer_class, self._route, str(status_code)).inc()
        self._metrics.duration.labels(self._consumer_class, self._route).observe(
            time.perf_counter() - self._started
        )


def create_app(
    config: SidecarConfig,
    keyring: ConsumerKeyring | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy application for one engine pod.

    The keyring is loaded once here rather than per request: rotating a key is a
    pod restart, which is already how a projected Secret is rolled, and it keeps
    an unreadable file from being discovered halfway through a shift.
    """

    resolved = keyring if keyring is not None else load_keyring(config.keys_file, config.bands)
    verdict = config.sizing_verdict()
    if verdict is not None and not verdict.sequence_limit_binds_first:
        raise ValueError(f"unsafe model-serving configuration: {verdict.detail}")
    if verdict is not None:
        logger.info(
            "kv budget %.1f GiB, %s",
            verdict.kv_budget_bytes / BYTES_PER_GIB,
            verdict.detail,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Hold one upstream client for the process lifetime."""

        async with httpx.AsyncClient(
            base_url=config.upstream_base_url,
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(
        title="FlakeGraph inference sidecar",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.keyring = resolved
    metrics = SidecarMetrics()
    app.state.metrics = metrics

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy(path: str, request: Request) -> Response:
        """Authenticate, stamp priority, and relay the request to the engine."""

        route = f"/{path}"
        if route in REFUSED_PATHS:
            refused = _Accounting(metrics, UNAUTHENTICATED_CLASS, route)
            return refused.close(
                JSONResponse({"error": "adapter management is disabled"}, status_code=404)
            )

        if route == "/metrics":
            return await _scrape(request.app.state.client, metrics)
        body = await request.body()
        # Probe and scoring traffic is not measured. The kubelet and the picker
        # call these paths many times a second, which would bury the consumer
        # series in a volume of requests that never reach the scheduler.
        if route in UNAUTHENTICATED_PATHS:
            return await _relay(request, route, body, None)

        consumer_class = resolved.classify(_presented_key(request))
        if consumer_class is None:
            rejected = _Accounting(metrics, UNAUTHENTICATED_CLASS, route)
            return rejected.close(JSONResponse({"error": "unauthorized"}, status_code=401))

        accounting = _Accounting(metrics, consumer_class, route)
        stamped, priority = _stamped_body(body, route, resolved, consumer_class)
        if priority is not None:
            metrics.priority_stamped.labels(consumer_class, str(priority)).inc()
        return await _relay(request, route, stamped, accounting)

    async def _relay(
        request: Request,
        route: str,
        body: bytes,
        accounting: _Accounting | None,
    ) -> Response:
        """Stream the upstream response back without buffering the whole body."""

        client: httpx.AsyncClient = request.app.state.client
        upstream = client.build_request(
            request.method,
            route,
            params=request.query_params,
            headers=_forwarded_headers(request.headers),
            content=body,
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.ConnectError:
            # Normal for the whole of a cold start: the engine spends minutes
            # loading weights before it binds its port, and the kubelet is
            # probing this path throughout. Report it as a bad gateway so the
            # probe reads "not ready" instead of the process logging a stack
            # trace per second until the engine appears.
            unavailable = JSONResponse({"error": "engine unavailable"}, status_code=502)
            return accounting.close(unavailable) if accounting is not None else unavailable
        except BaseException:
            if accounting is not None:
                accounting.abandon()
            raise
        chunks = response.aiter_raw()
        if accounting is not None:
            chunks = accounting.stream(chunks, response.status_code)
        return StreamingResponse(
            chunks,
            status_code=response.status_code,
            headers=_relayed_headers(response.headers),
            background=BackgroundTask(response.aclose),
        )

    return app


async def _scrape(client: httpx.AsyncClient, metrics: SidecarMetrics) -> Response:
    """Answer one scrape with the engine's series followed by the sidecar's.

    The engine's are the ones worth having, so they are relayed rather than
    replaced - but the sidecar's are appended even when the engine cannot be
    reached, because "the sidecar is up" must not read as "down" for the whole
    of a cold start. The scraper's own headers are deliberately not forwarded:
    an ``Accept`` asking for OpenMetrics would make the engine terminate its
    body with ``# EOF``, after which nothing may follow.
    """

    own = metrics.render()
    try:
        upstream = await client.get("/metrics", timeout=SCRAPE_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return Response(own, media_type=CONTENT_TYPE_LATEST)
    if not upstream.is_success:
        return Response(own, media_type=CONTENT_TYPE_LATEST)
    engine = upstream.content
    if engine and not engine.endswith(b"\n"):
        engine += b"\n"
    return Response(engine + own, media_type=CONTENT_TYPE_LATEST)


def run(config: SidecarConfig | None = None) -> None:
    """Serve the sidecar until the process is signalled."""

    resolved = config if config is not None else SidecarConfig.from_env()
    uvicorn.run(
        create_app(resolved),
        host=resolved.listen_host,
        port=resolved.listen_port,
        log_level="info",
    )


def _presented_key(request: Request) -> str:
    """Extract a bearer credential without treating a malformed header as valid."""

    header = request.headers.get("authorization", "")
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return credential.strip()


def _stamped_body(
    body: bytes,
    route: str,
    keyring: ConsumerKeyring,
    consumer_class: str,
) -> tuple[bytes, int | None]:
    """Remove any client priority and stamp the server's band for this class.

    Returns the body to relay and the band stamped into it, or ``None`` when
    none was. A body that is not a JSON object is relayed untouched. It cannot
    carry a priority field, so there is nothing to strip and nothing to forge.
    """

    if not body:
        return body, None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body, None
    if not isinstance(payload, dict):
        return body, None
    payload.pop(PRIORITY_FIELD, None)
    priority: int | None = None
    if route in STAMPED_PATHS:
        priority = keyring.priority_for(consumer_class)
        payload[PRIORITY_FIELD] = priority
    return json.dumps(payload).encode("utf-8"), priority


def _forwarded_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop connection-scoped headers, the caller's credential, and priority hints.

    The engine has no authentication of its own, so relaying the client's key
    would only write it into another process's logs. Length is dropped because
    stamping changes it, and httpx recomputes it from the body it is given.
    """

    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in HOP_BY_HOP_HEADERS or lowered in PRIORITY_HEADERS:
            continue
        if lowered in {"host", "content-length", "authorization"}:
            continue
        forwarded[name] = value
    return forwarded


def _relayed_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return upstream response headers minus the connection-scoped ones.

    The raw byte stream is relayed unchanged, so any content encoding and the
    length that describes it both remain accurate.
    """

    return {
        name: value for name, value in headers.items() if name.lower() not in HOP_BY_HOP_HEADERS
    }


def _optional_model[ModelT: BaseModel](
    model: type[ModelT],
    env: Mapping[str, str],
    variable: str,
) -> ModelT | None:
    """Parse a JSON-valued environment variable into a model when it is present."""

    raw = env.get(variable)
    if raw is None or not raw.strip():
        return None
    return model.model_validate_json(raw)
