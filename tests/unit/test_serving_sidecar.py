from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from kg_processor.serving.priority import ConsumerKeyring, load_keyring
from kg_processor.serving.sidecar import SidecarConfig, create_app
from kg_processor.serving.sizing import DeviceBudget, ModelGeometry

KEYRING = ConsumerKeyring(
    bands={"interactive": 0, "dev": 10, "batch": 100},
    keys={"sk-chat": "interactive", "sk-tool": "dev", "sk-pipeline": "batch"},
)


def _stub(body: bytes, content_type: str) -> httpx.Response:
    """Build an unread response so the sidecar can relay it as a raw stream.

    ``httpx.Response`` reads eager content during construction, which would leave
    nothing for ``aiter_raw`` to iterate. A real transport always hands back an
    unconsumed stream, so the double has to as well.
    """

    return httpx.Response(
        200,
        headers={"content-type": content_type, "content-length": str(len(body))},
        stream=httpx.ByteStream(body),
    )


class _Upstream:
    """Records what the engine would have received and replies with a stub."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/metrics":
            return _stub(b"vllm:num_preemptions_total 0", "text/plain")
        return _stub(b'{"ok": true}', "application/json")

    @property
    def last_body(self) -> dict[str, object]:
        payload: dict[str, object] = json.loads(self.requests[-1].content)
        return payload


def _client(upstream: _Upstream, config: SidecarConfig | None = None) -> TestClient:
    resolved = config or SidecarConfig(upstream_base_url="http://engine.invalid")
    app = create_app(resolved, keyring=KEYRING, transport=httpx.MockTransport(upstream.handler))
    return TestClient(app)


def test_batch_key_cannot_forge_the_interactive_band() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-pipeline"},
            json={"model": "m", "messages": [], "priority": 0},
        )

    assert response.status_code == 200
    assert upstream.last_body["priority"] == 100


def test_each_class_is_stamped_with_its_own_band() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        for key, expected in (("sk-chat", 0), ("sk-tool", 10), ("sk-pipeline", 100)):
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "m", "messages": []},
            )
            assert upstream.last_body["priority"] == expected


def test_priority_headers_never_reach_the_engine() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer sk-pipeline", "X-Vllm-Priority": "0"},
            json={"model": "m", "prompt": "hi"},
        )

    forwarded = upstream.requests[-1].headers
    assert "x-vllm-priority" not in forwarded
    assert "authorization" not in forwarded


def test_priority_is_stripped_even_on_paths_that_are_not_stamped() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        client.post(
            "/tokenize",
            headers={"Authorization": "Bearer sk-pipeline"},
            json={"model": "m", "prompt": "hi", "priority": 0},
        )

    assert "priority" not in upstream.last_body


def test_an_unknown_key_is_rejected_before_the_engine_is_touched() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-not-issued"},
            json={"model": "m", "messages": []},
        )

    assert response.status_code == 401
    assert upstream.requests == []


def test_a_missing_credential_is_rejected() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        assert client.post("/v1/chat/completions", json={}).status_code == 401
        assert client.get("/v1/models").status_code == 401


def test_an_unknown_class_receives_the_band_served_last() -> None:
    upstream = _Upstream()
    keyring = ConsumerKeyring(
        bands={"interactive": 0, "batch": 100},
        keys={"sk-mystery": "reporting"},
    )
    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=keyring,
        transport=httpx.MockTransport(upstream.handler),
    )
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-mystery"},
            json={"model": "m", "messages": []},
        )

    assert upstream.last_body["priority"] == 100


def test_probe_and_scoring_paths_stay_reachable_without_a_key() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_adapter_management_is_refused_outright() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        response = client.post(
            "/load_lora_adapter",
            headers={"Authorization": "Bearer sk-chat"},
            json={"lora_name": "x", "lora_path": "/tmp/x"},
        )

    assert response.status_code == 404
    assert upstream.requests == []


def test_streamed_completions_pass_through_chunk_by_chunk() -> None:
    chunks = [b'data: {"a":1}\n\n', b'data: {"a":2}\n\n', b"data: [DONE]\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(chunks)),
        )

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client, client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-chat"},
        json={"model": "m", "messages": [], "stream": True},
    ) as response:
        assert response.headers["content-type"] == "text/event-stream"
        assert b"".join(response.iter_bytes()) == b"".join(chunks)


def test_a_cold_engine_reads_as_not_ready_rather_than_crashing() -> None:
    """A probe during weight loading must get a clean 502, not a stack trace."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    app = create_app(
        SidecarConfig(upstream_base_url="http://engine.invalid"),
        keyring=KEYRING,
        transport=httpx.MockTransport(refuse),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 502
        assert (
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-chat"},
                json={"model": "m", "messages": []},
            ).status_code
            == 502
        )


def test_query_parameters_and_method_survive_the_hop() -> None:
    upstream = _Upstream()
    with _client(upstream) as client:
        client.get("/v1/models", params={"limit": "5"}, headers={"Authorization": "Bearer sk-chat"})

    assert upstream.requests[-1].method == "GET"
    assert upstream.requests[-1].url.params["limit"] == "5"


def test_an_unsafe_sequence_limit_prevents_the_app_from_being_built() -> None:
    config = SidecarConfig(
        upstream_base_url="http://engine.invalid",
        geometry=ModelGeometry(
            kv_heads=4,
            head_dim=256,
            attention_layers=16,
            weights_bytes=23_416_000_000,
        ),
        budget=DeviceBudget(
            device_memory_bytes=127_990_000_000,
            gpu_memory_utilization=0.60,
            overhead_bytes=12_884_000_000,
        ),
        expected_context_tokens=32768,
        max_num_seqs=512,
    )

    with pytest.raises(ValueError, match="does not bind before KV memory"):
        create_app(config, keyring=KEYRING)


def test_the_keyring_is_read_from_the_projected_secret(tmp_path: Path) -> None:
    keys_file = tmp_path / "serving-keys.json"
    keys_file.write_text(json.dumps({"sk-a": "interactive"}), encoding="utf-8")

    keyring = load_keyring(keys_file, {"interactive": 0, "batch": 100})

    assert keyring.classify("sk-a") == "interactive"
    assert keyring.classify("sk-b") is None
    assert keyring.priority_for(None) == 100


def test_sizing_inputs_must_be_configured_as_a_pair() -> None:
    env = {
        "FLAKEGRAPH_SIDECAR_GEOMETRY": json.dumps(
            {"kv_heads": 4, "head_dim": 256, "attention_layers": 16, "weights_bytes": 1}
        )
    }

    with pytest.raises(ValueError, match="must be set together"):
        SidecarConfig.from_env(env)
