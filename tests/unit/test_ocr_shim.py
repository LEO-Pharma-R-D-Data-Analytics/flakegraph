from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import psycopg
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from psycopg_pool import AsyncConnectionPool
from starlette.requests import ClientDisconnect

from kg_processor.serving import ocr_shim
from kg_processor.serving.ocr_shim import (
    OcrQueue,
    OcrShimConfig,
    QueueDepth,
    UpstreamPool,
    create_app,
)
from kg_processor.serving.priority import ConsumerKeyring

KEYRING = ConsumerKeyring(
    bands={"interactive": 0, "dev": 10, "batch": 100},
    keys={"sk-chat": "interactive", "sk-pipeline": "batch"},
)


class _RecordingQueue:
    """Stands in for PostgreSQL, recording the order work was admitted in.

    Admission sends every request to the last replica offered, so a test can
    tell the queue's choice apart from the pool's own ordering.
    """

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, int, str]] = []
        self.admitted: list[str] = []
        self.renewed: list[tuple[str, str, str | None]] = []
        self.released: list[str] = []
        self.offered: list[tuple[tuple[str, ...], int]] = []
        self.admit = True
        self.renew_failures = 0

    async def enqueue(self, request_id: str, priority: int, consumer_class: str) -> None:
        self.enqueued.append((request_id, priority, consumer_class))

    async def try_admit(
        self, request_id: str, replicas: tuple[str, ...], capacity_per_replica: int
    ) -> str | None:
        self.offered.append((replicas, capacity_per_replica))
        if not self.admit:
            return None
        self.admitted.append(request_id)
        return replicas[-1]

    async def renew(
        self,
        request_id: str,
        priority: int,
        consumer_class: str,
        status: str,
        replica: str | None = None,
    ) -> None:
        if self.renew_failures:
            self.renew_failures -= 1
            raise psycopg.OperationalError("the queue database is failing over")
        self.renewed.append((request_id, status, replica))

    async def release(self, request_id: str) -> None:
        self.released.append(request_id)

    async def depth(self) -> list[QueueDepth]:
        return []


class _FakeConnectionPool:
    """Stands in for the psycopg pool, answering every query with fixed rows.

    Either hands out a connection whose cursors return ``rows``, or refuses to
    hand out any connection at all when built with ``failure``.
    """

    def __init__(
        self, rows: list[dict[str, Any]] | None = None, failure: Exception | None = None
    ) -> None:
        self.rows = rows or []
        self.failure = failure

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[_FakeConnectionPool]:
        if self.failure is not None:
            raise self.failure
        yield self

    async def execute(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> _FakeConnectionPool:
        return self

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


def _samples(exposition: str, name: str) -> dict[tuple[tuple[str, str], ...], float]:
    """Return every sample of one series, keyed by its sorted label pairs."""

    return {
        tuple(sorted(sample.labels.items())): float(sample.value)
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    }


def _pool(replicas: tuple[str, ...], capacity: int = 2) -> UpstreamPool:
    async def resolver() -> tuple[str, ...]:
        return replicas

    return UpstreamPool("mineru.invalid", 8080, capacity, resolver=resolver)


def _stub(body: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/json", "content-length": str(len(body))},
        stream=httpx.ByteStream(body),
    )


class _Pool:
    """Records which replica each request reached."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return _stub(b'{"results": {}}')


class _BrokenStream(httpx.AsyncByteStream):
    """A body whose connection drops after its first chunk."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"resu'
        raise httpx.ReadError("connection reset by peer")


class _CollapsingPool(_Pool):
    """A replica that answers and then dies mid-body."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, stream=_BrokenStream())


class _SlowPool(_Pool):
    """A replica whose parse takes longer than the stale window."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    async def handler(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        self.requests.append(request)
        await asyncio.sleep(self.seconds)
        return _stub(b'{"results": {}}')

    @property
    def hosts(self) -> list[str]:
        return [request.url.host for request in self.requests]


def _client(
    upstream: _Pool,
    queue: _RecordingQueue | OcrQueue,
    replicas: tuple[str, ...] = ("10.0.0.1", "10.0.0.2"),
    capacity: int = 2,
    stale_after_seconds: float = 60.0,
) -> TestClient:
    app = create_app(
        OcrShimConfig(
            database_url="postgresql://unused",
            upstream_host="mineru.invalid",
            poll_interval_seconds=0.01,
            stale_after_seconds=stale_after_seconds,
        ),
        keyring=KEYRING,
        transport=httpx.MockTransport(upstream.handler),
        queue=queue,  # type: ignore[arg-type]
        upstreams=_pool(replicas, capacity),
    )
    return TestClient(app)


def test_a_request_is_queued_at_the_band_its_key_holds() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-pipeline"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert [entry[1:] for entry in queue.enqueued] == [(100, "batch")]


def test_interactive_and_batch_keys_queue_at_different_bands() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        for key in ("sk-chat", "sk-pipeline"):
            client.post(
                "/file_parse",
                headers={"Authorization": f"Bearer {key}"},
                files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            )

    assert [entry[1] for entry in queue.enqueued] == [0, 100]


def test_an_unknown_key_never_reaches_the_queue_or_the_pool() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post("/file_parse", headers={"Authorization": "Bearer sk-nope"})

    assert response.status_code == 401
    assert queue.enqueued == []
    assert upstream.requests == []


def test_admission_is_offered_every_replica_the_shim_resolves() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1", "10.0.0.2", "10.0.0.3"), capacity=4) as c:
        c.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert queue.offered[0] == (("10.0.0.1", "10.0.0.2", "10.0.0.3"), 4)


def test_the_request_goes_to_the_replica_the_queue_chose() -> None:
    """Placement is the queue's decision, made from the fleet's load, not this shim's."""

    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1", "10.0.0.2")) as client:
        client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        exposition = client.get("/metrics").text

    assert upstream.requests[0].url.host == "10.0.0.2"
    assert _samples(exposition, "flakegraph_ocr_parse_duration_seconds_count") == {
        (("replica", "10.0.0.2"),): 1
    }


def test_the_callers_credential_is_not_relayed_to_the_parsing_pool() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert "authorization" not in upstream.requests[-1].headers


def test_a_finished_request_frees_its_slot() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 200

    assert queue.released == queue.admitted


def test_a_caller_that_hangs_up_while_queued_is_not_parsed_for() -> None:
    """The worker re-sends after its timeout, so the orphan would only run twice."""

    queue = _RecordingQueue()
    queue.admit = False
    with _client(_Pool(), queue) as client:
        app = client.app

        async def hung_up() -> dict[str, str]:
            return {"type": "http.disconnect"}

        request = Request({"type": "http", "method": "POST", "headers": [], "app": app}, hung_up)
        upstreams, metrics = app.state.upstreams, app.state.metrics
        config = OcrShimConfig(
            database_url="postgresql://unused",
            upstream_host="mineru.invalid",
            poll_interval_seconds=0.01,
        )
        with pytest.raises(ClientDisconnect):
            asyncio.run(
                ocr_shim._hold_and_forward(
                    request, "/file_parse", b"", 0, "interactive", upstreams, config, metrics
                )
            )

    assert queue.released == [entry[0] for entry in queue.enqueued]
    assert queue.admitted == []
    assert _samples(metrics.render().body.decode(), "flakegraph_ocr_requests_total") == {
        (("consumer_class", "interactive"), ("outcome", "client_gone")): 1
    }


def test_a_parse_longer_than_the_stale_window_stays_counted_as_busy() -> None:
    upstream, queue = _SlowPool(seconds=0.2), _RecordingQueue()
    with _client(upstream, queue, stale_after_seconds=0.06) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 200

    # Admission was immediate, so every renewal happened while MinerU was busy,
    # and each one carries enough to put the row back should a sweep take it.
    assert queue.renewed and {entry[1:] for entry in queue.renewed} == {("dispatched", "10.0.0.2")}
    assert [entry[0] for entry in queue.renewed] == queue.admitted * len(queue.renewed)
    assert queue.released == queue.admitted


def test_a_failed_renewal_does_not_end_the_heartbeat(caplog: pytest.LogCaptureFixture) -> None:
    """The parse keeps running whether or not the queue database answered."""

    upstream, queue = _SlowPool(seconds=0.2), _RecordingQueue()
    queue.renew_failures = 1
    with caplog.at_level(logging.WARNING), _client(upstream, queue, stale_after_seconds=0.06) as c:
        response = c.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 200

    assert "could not renew dispatched request" in caplog.text
    assert queue.renewed


def test_a_replica_dying_mid_body_still_frees_its_slot() -> None:
    upstream, queue = _CollapsingPool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1",)) as client:
        with pytest.raises(httpx.ReadError):
            client.post(
                "/file_parse",
                headers={"Authorization": "Bearer sk-chat"},
                files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            )
        exposition = client.get("/metrics").text

    assert queue.released == queue.admitted
    assert _samples(exposition, "flakegraph_ocr_in_flight") == {(("replica", "10.0.0.1"),): 0}
    assert _samples(exposition, "flakegraph_ocr_requests_total") == {
        (("consumer_class", "interactive"), ("outcome", "upstream_error")): 1
    }


def test_health_reports_the_replicas_the_shim_can_currently_see() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1", "10.0.0.2")) as client:
        payload = client.get("/health").json()

    assert payload == {"status": "ok", "replicas": 2}


def test_the_replicas_offered_follow_the_pool_as_it_scales() -> None:
    replicas: list[tuple[str, ...]] = [("10.0.0.1",)]

    async def resolver() -> tuple[str, ...]:
        return replicas[0]

    pool = UpstreamPool("mineru.invalid", 8080, 8, resolver=resolver)
    asyncio.run(pool.refresh())
    assert pool.endpoints == ("10.0.0.1",)

    replicas[0] = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    asyncio.run(pool.refresh())
    assert pool.endpoints == ("10.0.0.1", "10.0.0.2", "10.0.0.3")


def test_an_ipv6_replica_is_addressed_with_brackets() -> None:
    pool = _pool(("fd00::1",))

    assert pool.base_url("fd00::1") == "http://[fd00::1]:8080"
    assert pool.base_url("10.0.0.1") == "http://10.0.0.1:8080"


def test_configuration_is_read_from_the_pods_environment() -> None:
    config = OcrShimConfig.from_env(
        {
            "FLAKEGRAPH_OCR_SHIM_DATABASE_URL": "postgresql://db/flakegraph",
            "FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST": "release-flakegraph-mineru",
            "FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY": "8",
            "FLAKEGRAPH_OCR_SHIM_BANDS": json.dumps({"interactive": 0, "batch": 100}),
        }
    )

    assert config.upstream_capacity == 8
    assert config.bands == {"interactive": 0, "batch": 100}


def test_a_shim_without_a_database_url_refuses_to_be_configured() -> None:
    with pytest.raises(ValueError, match="database_url"):
        OcrShimConfig.from_env({"FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST": "mineru"})


def test_a_parse_that_omits_a_backend_is_given_the_one_the_pool_supports() -> None:
    """MinerU's own default is a backend this image cannot run.

    The pool installs the pipeline extra only, so `hybrid-engine` fails - but not
    before downloading a VLM, which turns a configuration mistake into several
    wasted minutes and an opaque 409. A caller who follows MinerU's documentation
    and omits the field should still be served.
    """

    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("paper.pdf", b"%PDF-1.7 body", "application/pdf")},
            data={"return_md": "true"},
        )

    assert response.status_code == 200
    forwarded = upstream.requests[0].content
    assert b'name="backend"' in forwarded
    assert b"pipeline" in forwarded
    # The upload itself must survive the rewrite untouched.
    assert b"%PDF-1.7 body" in forwarded
    assert b'name="return_md"' in forwarded


def test_a_backend_the_pool_cannot_run_is_refused_before_it_costs_anything() -> None:
    """Refusing here is the difference between an answer and a five-minute wait."""

    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("paper.pdf", b"%PDF-1.7 body", "application/pdf")},
            data={"backend": "hybrid-engine"},
        )

    assert response.status_code == 400
    assert "hybrid-engine" in response.json()["error"]
    assert response.json()["supported_backends"] == ["pipeline"]
    # Nothing was queued and the pool was never touched.
    assert upstream.requests == []
    assert queue.enqueued == []


def test_an_explicit_supported_backend_is_left_exactly_as_the_caller_sent_it() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("paper.pdf", b"%PDF-1.7 body", "application/pdf")},
            data={"backend": "pipeline"},
        )

    assert response.status_code == 200
    forwarded = upstream.requests[0].content
    assert forwarded.count(b'name="backend"') == 1


def test_a_file_containing_the_word_backend_is_not_mistaken_for_the_field() -> None:
    """The field is read from each part's own headers, not searched for."""

    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={
                "files": (
                    "paper.pdf",
                    b'a paper about name="backend" choices',
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200
    forwarded = upstream.requests[0].content
    # The default was still supplied, because the document is not the field.
    assert forwarded.count(b'name="backend"') == 2
    assert b"pipeline" in forwarded


def test_the_queue_pool_checks_a_connection_before_handing_it_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection the server killed while idle is replaced, not served.

    Without a checkout check the pool returns the dead connection and the
    request that draws it fails with the server's termination message - once,
    and for nothing the caller did.
    """

    captured: dict[str, Any] = {}

    class _RecordingPool:
        check_connection = AsyncConnectionPool.check_connection

        def __init__(self, conninfo: str, **kwargs: Any) -> None:
            captured["conninfo"] = conninfo
            captured.update(kwargs)

        async def __aenter__(self) -> _RecordingPool:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(ocr_shim, "AsyncConnectionPool", _RecordingPool)
    app = create_app(
        OcrShimConfig(
            database_url="postgresql://queue.invalid/flakegraph", upstream_host="mineru.invalid"
        ),
        keyring=KEYRING,
        transport=httpx.MockTransport(_Pool().handler),
        upstreams=_pool(("10.0.0.1",), 1),
    )
    with TestClient(app):
        pass

    assert captured["conninfo"] == "postgresql://queue.invalid/flakegraph"
    assert captured["check"] is AsyncConnectionPool.check_connection


def test_metrics_is_a_text_exposition_of_the_shims_own_series() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = {family.name for family in text_string_to_metric_families(response.text)}
    assert {
        "flakegraph_ocr_requests",
        "flakegraph_ocr_queue_wait_seconds",
        "flakegraph_ocr_parse_duration_seconds",
        "flakegraph_ocr_in_flight",
        "flakegraph_ocr_replica_capacity",
        "flakegraph_ocr_upstream_replicas",
    } <= families
    # The ping route stays a liveness answer, not a scrape.
    assert client.get("/ping").json() == {"status": "ok"}


def test_a_successful_parse_is_observed_against_the_replica_that_served_it() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1",), capacity=3) as client:
        response = client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 200
        exposition = client.get("/metrics").text

    assert _samples(exposition, "flakegraph_ocr_requests_total") == {
        (("consumer_class", "interactive"), ("outcome", "succeeded")): 1
    }
    assert _samples(exposition, "flakegraph_ocr_queue_wait_seconds_count") == {
        (("consumer_class", "interactive"),): 1
    }
    assert _samples(exposition, "flakegraph_ocr_parse_duration_seconds_count") == {
        (("replica", "10.0.0.1"),): 1
    }
    assert _samples(exposition, "flakegraph_ocr_in_flight") == {(("replica", "10.0.0.1"),): 0}
    assert _samples(exposition, "flakegraph_ocr_replica_capacity") == {
        (("replica", "10.0.0.1"),): 3
    }
    assert _samples(exposition, "flakegraph_ocr_upstream_replicas") == {(): 1}


def test_a_refused_request_is_counted_without_reaching_the_queue() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue) as client:
        client.post("/file_parse", headers={"Authorization": "Bearer sk-nope"})
        client.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            data={"backend": "hybrid-engine"},
        )
        exposition = client.get("/metrics").text

    assert _samples(exposition, "flakegraph_ocr_requests_total") == {
        (("consumer_class", "unauthenticated"), ("outcome", "rejected")): 1,
        (("consumer_class", "interactive"), ("outcome", "rejected")): 1,
    }
    assert "sk-nope" not in exposition


def test_queue_depth_is_read_across_the_whole_fleet_at_scrape_time() -> None:
    rows = [
        {"consumer_class": "batch", "status": "waiting", "depth": 3, "oldest_seconds": 42.5},
        {"consumer_class": "batch", "status": "dispatched", "depth": 2, "oldest_seconds": 90.0},
        {"consumer_class": "interactive", "status": "waiting", "depth": 1, "oldest_seconds": 0.5},
    ]
    queue = OcrQueue(_FakeConnectionPool(rows), "shim-a", 60.0)
    with _client(_Pool(), queue) as client:
        exposition = client.get("/metrics").text

    # Every class the keyring knows is present in every status, at zero when
    # the queue holds nothing of it: a class with no work must read as empty,
    # not as missing.
    assert _samples(exposition, "flakegraph_ocr_queue_depth") == {
        (("consumer_class", "interactive"), ("status", "waiting")): 1,
        (("consumer_class", "interactive"), ("status", "dispatched")): 0,
        (("consumer_class", "dev"), ("status", "waiting")): 0,
        (("consumer_class", "dev"), ("status", "dispatched")): 0,
        (("consumer_class", "batch"), ("status", "waiting")): 3,
        (("consumer_class", "batch"), ("status", "dispatched")): 2,
    }
    # Only what is still waiting has an age worth alerting on.
    assert _samples(exposition, "flakegraph_ocr_oldest_waiting_seconds") == {
        (("consumer_class", "interactive"),): 0.5,
        (("consumer_class", "dev"),): 0.0,
        (("consumer_class", "batch"),): 42.5,
    }


def test_an_empty_queue_reads_as_zero_for_every_class() -> None:
    queue = OcrQueue(_FakeConnectionPool([]), "shim-a", 60.0)
    with _client(_Pool(), queue) as client:
        exposition = client.get("/metrics").text

    depth = _samples(exposition, "flakegraph_ocr_queue_depth")
    assert set(depth.values()) == {0}
    assert {labels[0][1] for labels in depth} == {"interactive", "dev", "batch"}
    assert set(_samples(exposition, "flakegraph_ocr_oldest_waiting_seconds").values()) == {0.0}


def test_a_scrape_survives_the_queue_database_being_unreachable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = psycopg.OperationalError("connection refused")
    queue = OcrQueue(_FakeConnectionPool(failure=failure), "shim-a", 60.0)
    with _client(_Pool(), queue) as client, caplog.at_level(logging.WARNING):
        response = client.get("/metrics")

    assert response.status_code == 200
    families = {family.name for family in text_string_to_metric_families(response.text)}
    assert "flakegraph_ocr_requests" in families
    assert "flakegraph_ocr_replica_capacity" in families
    assert "flakegraph_ocr_queue_depth" not in families
    assert "queue depth is unavailable" in caplog.text
