from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from kg_processor.serving.ocr_shim import OcrShimConfig, UpstreamPool, create_app
from kg_processor.serving.priority import ConsumerKeyring

KEYRING = ConsumerKeyring(
    bands={"interactive": 0, "dev": 10, "batch": 100},
    keys={"sk-chat": "interactive", "sk-pipeline": "batch"},
)


class _RecordingQueue:
    """Stands in for PostgreSQL, recording the order work was admitted in."""

    def __init__(self, capacity_seen: list[int] | None = None) -> None:
        self.enqueued: list[tuple[str, int, str]] = []
        self.admitted: list[str] = []
        self.released: list[str] = []
        self.capacity_seen = capacity_seen if capacity_seen is not None else []
        self.admit = True

    async def enqueue(self, request_id: str, priority: int, consumer_class: str) -> None:
        self.enqueued.append((request_id, priority, consumer_class))

    async def try_admit(self, request_id: str, capacity: int) -> bool:
        self.capacity_seen.append(capacity)
        if not self.admit:
            return False
        self.admitted.append(request_id)
        return True

    async def renew(self, request_id: str) -> None:
        return None

    async def release(self, request_id: str) -> None:
        self.released.append(request_id)


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

    @property
    def hosts(self) -> list[str]:
        return [request.url.host for request in self.requests]


def _client(
    upstream: _Pool,
    queue: _RecordingQueue,
    replicas: tuple[str, ...] = ("10.0.0.1", "10.0.0.2"),
    capacity: int = 2,
) -> TestClient:
    app = create_app(
        OcrShimConfig(
            database_url="postgresql://unused",
            upstream_host="mineru.invalid",
            poll_interval_seconds=0.01,
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


def test_admission_is_offered_the_capacity_the_whole_pool_has() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1", "10.0.0.2", "10.0.0.3"), capacity=4) as c:
        c.post(
            "/file_parse",
            headers={"Authorization": "Bearer sk-chat"},
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert queue.capacity_seen[0] == 12


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


def test_health_reports_the_replicas_the_shim_can_currently_see() -> None:
    upstream, queue = _Pool(), _RecordingQueue()
    with _client(upstream, queue, replicas=("10.0.0.1", "10.0.0.2")) as client:
        payload = client.get("/health").json()

    assert payload == {"status": "ok", "replicas": 2}


def test_dispatch_prefers_the_least_loaded_replica() -> None:
    pool = _pool(("10.0.0.1", "10.0.0.2"), capacity=2)
    asyncio.run(pool.refresh())

    first = pool.acquire()
    second = pool.acquire()
    third = pool.acquire()
    fourth = pool.acquire()

    # Two replicas at two each: every slot is used exactly once before any
    # replica takes a second request.
    assert {first, second} == {"10.0.0.1", "10.0.0.2"}
    assert {third, fourth} == {"10.0.0.1", "10.0.0.2"}
    assert pool.acquire() is None


def test_a_released_slot_becomes_available_again() -> None:
    pool = _pool(("10.0.0.1",), capacity=1)
    asyncio.run(pool.refresh())

    assert pool.acquire() == "10.0.0.1"
    assert pool.acquire() is None
    pool.release("10.0.0.1")
    assert pool.acquire() == "10.0.0.1"


def test_capacity_follows_the_pool_as_it_scales() -> None:
    replicas: list[tuple[str, ...]] = [("10.0.0.1",)]

    async def resolver() -> tuple[str, ...]:
        return replicas[0]

    pool = UpstreamPool("mineru.invalid", 8080, 8, resolver=resolver)
    asyncio.run(pool.refresh())
    assert pool.capacity == 8

    replicas[0] = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    asyncio.run(pool.refresh())
    assert pool.capacity == 24


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
