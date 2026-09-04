"""Hold the one-gate contract: in front of the app, and not across its paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from flakegraph_app.ui import authentication

_CHART = Path("deploy/helm/flakegraph")
_AUTH_PROXY_TEMPLATE = _CHART / "templates/auth-proxy.yaml"
_CONTROL_PLANE_TEMPLATE = _CHART / "templates/control-plane.yaml"


def test_the_gate_claims_only_its_own_path_segment() -> None:
    """Leave application paths that merely start with `/oauth2` alone.

    Traefik prefix-matches the raw string, so a rule for `/oauth2` also captures
    `/oauth2callback` - an application's own OIDC callback. Routing that to the
    gate answers the end of a sign-in with the start of one, for ever.
    """

    template = _AUTH_PROXY_TEMPLATE.read_text(encoding="utf-8")

    assert "- path: /oauth2/\n" in template
    assert "- path: /oauth2\n" not in template


def test_the_application_reads_the_gate_rather_than_signing_in_again() -> None:
    """One sign-in per viewer, performed by whichever layer is in front."""

    template = _CONTROL_PLANE_TEMPLATE.read_text(encoding="utf-8")

    assert "{{- if and .Values.ingress.enabled .Values.ingress.authProxy.enabled }}" in template
    assert "name: FLAKEGRAPH_APP_FORWARDED_IDENTITY_HEADER" in template
    assert "value: X-Auth-Request-Email" in template
    assert "/oauth2/sign_out" in template


def _streamlit_stub(headers: dict[str, str] | None) -> SimpleNamespace:
    def _refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("the application must not start a sign-in of its own")

    return SimpleNamespace(
        context=SimpleNamespace(headers=headers if headers is not None else {}),
        login=_refuse,
        user=SimpleNamespace(is_logged_in=False),
        secrets={},
    )


@pytest.fixture
def delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(authentication.FORWARDED_IDENTITY_ENVIRONMENT, "X-Auth-Request-Email")


def test_a_viewer_the_gate_admitted_is_served(
    delegated: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate already asked who this is; asking again is not more protection."""

    monkeypatch.setattr(
        authentication, "st", _streamlit_stub({"X-Auth-Request-Email": "someone@example.com"})
    )

    assert authentication.viewer_is_signed_in() is True
    assert authentication.require_sign_in(required=True) is True
    assert authentication.forwarded_identity() == "someone@example.com"


def test_a_request_that_bypassed_the_gate_is_refused(
    delegated: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reaching the application by a route the gate does not cover is not a login."""

    monkeypatch.setattr(authentication, "st", _streamlit_stub({}))

    with pytest.raises(authentication.AuthenticationNotConfigured):
        authentication.require_sign_in(required=True)


def test_naming_a_gate_makes_its_identity_mandatory(
    delegated: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`required` cannot relax a gate: naming one already demanded identity.

    The operator flag defaults to off, so honouring it here would serve every
    request that reaches this process by any route other than the gate.
    """

    monkeypatch.setattr(authentication, "st", _streamlit_stub({}))

    with pytest.raises(authentication.AuthenticationNotConfigured):
        authentication.require_sign_in(required=False)


def test_each_sign_in_flow_carries_its_own_csrf_nonce() -> None:
    """Survive a browser that asks for a favicon while the viewer signs in.

    With one fixed CSRF cookie name, every concurrently refused subresource
    overwrites the navigation's nonce, and the viewer returns from the issuer
    with a state the gate rejects as an attack.
    """

    template = _AUTH_PROXY_TEMPLATE.read_text(encoding="utf-8")

    assert "- --cookie-csrf-per-request=true" in template


def test_machine_interfaces_are_routed_without_the_browser_gate() -> None:
    """An SDK holding a bearer token cannot complete a browser sign-in.

    These paths keep their own authentication - a virtual key at the gateway, a
    keyring at the parsing shim - so routing them past the gate hands the check
    to the layer that can actually perform it.
    """

    values = yaml.safe_load((_CHART / "values.yaml").read_text(encoding="utf-8"))
    machine_paths = values["ingress"]["machineApiPaths"]

    # A namespace is never routed past the gate, only verified paths. The
    # gateway declares its key check per route, so `/v1` contains routes that
    # authenticate and routes that do not - `/v1/mcp/oauth/authorize` answers
    # anyone with a credential-entry page.
    assert "/v1" not in machine_paths["gateway"]
    assert not any(p.startswith("/v1/mcp") for p in machine_paths["gateway"])
    assert "/v1/chat/completions" in machine_paths["gateway"]
    assert machine_paths["ocr"] == ["/file_parse"]

    # Exactly, so a path cannot admit whatever upstream later adds beside it.
    assert "pathType: Exact" in (_CHART / "templates/ingress.yaml").read_text(encoding="utf-8")
    # The control plane is a browser application throughout: it has no interface
    # a program authenticates to, so nothing of it may leave the gate.
    assert "controlPlane" not in machine_paths

    template = (_CHART / "templates/ingress.yaml").read_text(encoding="utf-8")
    assert "machineApiPaths" in template
    # The gate's middleware is annotated onto one Ingress; the machine-API rules
    # are a second one precisely so they do not inherit it.
    assert template.count("router.middlewares") == 1


def test_the_parsing_shim_refuses_an_unauthenticated_parse() -> None:
    """Whatever leaves the gate must still refuse a caller with no credential."""

    shim = Path("src/kg_processor/serving/ocr_shim.py").read_text(encoding="utf-8")

    assert 'PARSE_ROUTE = "/file_parse"' in shim
    assert 'UNAUTHENTICATED_PATHS = frozenset({"/ping", "/metrics"})' in shim
    assert '"error": "unauthorized"}, status_code=401' in shim


def test_the_header_trusting_application_cannot_be_routed_past_the_gate() -> None:
    """The one service that does not authenticate its own callers must not leave.

    The control plane reads identity from a header the gate sets. Route it past
    the gate and that header stops being proof of anything - it becomes a request
    parameter any caller may set. Both the schema and the template refuse it.
    """

    schema = json.loads((_CHART / "values.schema.json").read_text(encoding="utf-8"))
    node = schema["properties"]["ingress"]["properties"]["machineApiPaths"]

    assert set(node["properties"]) == {"gateway", "ocr"}
    assert node["additionalProperties"] is False

    template = (_CHART / "templates/ingress.yaml").read_text(encoding="utf-8")
    assert 'hasKey ($ingress.machineApiPaths | default dict) "controlPlane"' in template
    assert "{{- fail " in template


def test_no_path_is_exempted_across_every_host_at_once() -> None:
    """A path exemption cannot tell which service is serving the path.

    Health endpoints are probed by the kubelet against the pod, never through
    the ingress, so exempting them protects nothing - and `^/health$` exempted
    for the parsing shim was also exempted for the control plane, where
    Streamlit answers it with the application shell.
    """

    values = yaml.safe_load((_CHART / "values.yaml").read_text(encoding="utf-8"))

    assert values["ingress"]["authProxy"]["skipAuthRoutes"] == []


def test_a_bare_namespace_cannot_be_routed_past_the_gate() -> None:
    """`/v1` is not a thing that authenticates; the paths under it are, or not."""

    schema = json.loads((_CHART / "values.schema.json").read_text(encoding="utf-8"))
    pattern = schema["properties"]["ingress"]["properties"]["machineApiPaths"]["properties"][
        "gateway"
    ]["items"]["pattern"]

    assert re.fullmatch(pattern, "/v1") is None
    assert re.fullmatch(pattern, "/v1/chat/completions") is not None
