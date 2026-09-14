"""Protect what the chart declares to a kube-prometheus-stack beside it."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

_CHART = Path("deploy/helm/flakegraph")
_VALUES = _CHART / "values.yaml"
_SCHEMA = _CHART / "values.schema.json"
_RULES_TEMPLATE = _CHART / "templates/monitoring-rules.yaml"
_NETWORK_POLICY_TEMPLATE = _CHART / "templates/model-serving-networkpolicy.yaml"
_POSTGRES_ADAPTER = Path("src/kg_processor/adapters/distributed/postgres.py")

_RELEASE = "fg"
_NAMESPACE = "fleet"
_FULLNAME = f"{_RELEASE}-flakegraph"
_API_VERSIONS = ",".join(
    f"monitoring.coreos.com/v1/{kind}"
    for kind in ("ServiceMonitor", "PodMonitor", "PrometheusRule")
)
# Everything the monitoring objects hang off: the database they read, the
# planes they scrape, and the ingress they publish Grafana through.
_ENABLED_SETTINGS = (
    "monitoring.enabled=true",
    "database.cloudNativePG.enabled=true",
    "modelServing.enabled=true",
    "modelServing.server.draftModelSeed.providedExternally=true",
    "ingress.enabled=true",
    "ingress.domain=example.test",
    "ingress.tls.secretName=wildcard-tls",
    "ingress.authProxy.enabled=true",
    "ingress.authProxy.existingSecret=oidc",
    "controlPlane.networkPolicy.enabled=true",
)
_MONITORING_KINDS = {"ServiceMonitor", "PodMonitor", "PrometheusRule"}


def test_monitoring_is_opt_in_and_leaves_no_trace_when_off() -> None:
    """Keep a cluster without the operator's CRDs installable by default.

    The objects need CRDs the chart does not ship, so the block is off, and
    off has to mean absent: not a ServiceMonitor, not a datasource, not the
    per-request callback in the gateway's config.
    """

    values = _load_yaml(_VALUES)["monitoring"]
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))

    assert values["enabled"] is False
    assert values["database"]["role"] == "flakegraph_metrics"
    assert "monitoring" in schema["required"]
    monitoring_schema = schema["properties"]["monitoring"]
    assert monitoring_schema["additionalProperties"] is False
    assert set(monitoring_schema["required"]) == set(values)
    thresholds_schema = monitoring_schema["properties"]["rules"]["properties"]["thresholds"]
    assert thresholds_schema["additionalProperties"] is False
    assert set(thresholds_schema["required"]) == set(values["rules"]["thresholds"])

    rendered = _render(())
    assert not [doc for doc in rendered if doc["kind"] in _MONITORING_KINDS]
    assert not [
        doc for doc in rendered if doc["metadata"].get("labels", {}).get("grafana_datasource")
    ]
    assert "callbacks" not in _gateway_config(rendered)


def test_monitoring_refuses_to_render_where_it_cannot_be_applied() -> None:
    """Name the missing prerequisite once, first, rather than per object at apply."""

    helm = _helm()
    without_crds = subprocess.run(
        [helm, "template", _RELEASE, str(_CHART), "--set", "monitoring.enabled=true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert without_crds.returncode != 0
    assert "monitoring.coreos.com/v1" in without_crds.stderr

    without_bundled_database = subprocess.run(
        [
            helm,
            "template",
            _RELEASE,
            str(_CHART),
            "--api-versions",
            _API_VERSIONS,
            "--set",
            "monitoring.enabled=true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert without_bundled_database.returncode != 0
    assert "database.cloudNativePG.enabled" in without_bundled_database.stderr


def test_every_plane_is_scraped_through_its_own_service() -> None:
    """Scrape each engine once, by name, and the picker with a token it can check."""

    rendered = _render(_ENABLED_SETTINGS)
    monitors = {doc["metadata"]["name"]: doc for doc in rendered if doc["kind"] == "ServiceMonitor"}
    interval = _load_yaml(_VALUES)["monitoring"]["scrapeInterval"]

    engines = monitors[f"{_FULLNAME}-vllm"]
    assert engines["spec"]["selector"]["matchLabels"]["flakegraph.io/scrape"] == "true"
    headless = _one(rendered, "Service", f"{_FULLNAME}-vllm-headless")
    balanced = _one(rendered, "Service", f"{_FULLNAME}-vllm")
    assert headless["metadata"]["labels"]["flakegraph.io/scrape"] == "true"
    assert "flakegraph.io/scrape" not in balanced["metadata"]["labels"]
    (endpoint,) = engines["spec"]["endpoints"]
    assert endpoint["port"] == "http"
    assert endpoint["path"] == "/metrics"
    assert endpoint["interval"] == interval
    relabelings = {rule["targetLabel"]: rule["sourceLabels"] for rule in endpoint["relabelings"]}
    assert relabelings["engine"] == ["__meta_kubernetes_pod_name"]
    assert relabelings["node"] == ["__meta_kubernetes_pod_node_name"]

    for name in (f"{_FULLNAME}-litellm", f"{_FULLNAME}-ocr"):
        (endpoint,) = monitors[name]["spec"]["endpoints"]
        assert endpoint["port"] == "http"
        assert endpoint["path"] == "/metrics"
        assert endpoint["interval"] == interval
        assert {"sourceLabels": ["__meta_kubernetes_pod_node_name"], "targetLabel": "node"} in (
            endpoint["relabelings"]
        )

    (endpoint,) = monitors[f"{_FULLNAME}-inference-router"]["spec"]["endpoints"]
    assert endpoint["port"] == "metrics"
    assert endpoint["bearerTokenFile"] == "/var/run/secrets/kubernetes.io/serviceaccount/token"
    router = _one(rendered, "Service", f"{_FULLNAME}-inference-router")
    ports = {port["name"]: port for port in router["spec"]["ports"]}
    assert ports["metrics"]["port"] == 9090
    assert ports["metrics"]["targetPort"] == "metrics"
    # Always on the Service: the endpoint authenticates, so it hands nobody
    # anything, and a ServiceMonitor can only name a Service port.
    assert "metrics" in {
        port["name"]
        for port in _one(_render(()), "Service", f"{_FULLNAME}-inference-router")["spec"]["ports"]
    }


def test_the_picker_is_allowed_to_evaluate_a_scrape() -> None:
    """Grant the two cluster-scoped reviews the metrics filter performs per request."""

    rendered = _render(_ENABLED_SETTINGS)
    name = f"{_FULLNAME}-inference-router-authn"

    role = _one(rendered, "ClusterRole", name)
    granted = {
        (group, resource)
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
        if "create" in rule["verbs"]
    }
    assert ("authentication.k8s.io", "tokenreviews") in granted
    assert ("authorization.k8s.io", "subjectaccessreviews") in granted
    binding = _one(rendered, "ClusterRoleBinding", name)
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "ClusterRole",
        "name": name,
    }
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": f"{_FULLNAME}-inference-router", "namespace": _NAMESPACE}
    ]
    assert not [
        doc
        for doc in _render(())
        if doc["kind"] == "ClusterRole" and doc["metadata"]["name"] == name
    ]


def test_prometheus_may_reach_the_engines_and_nothing_else_changes() -> None:
    """Admit the stack's Prometheus to the sidecar's port without widening the floor."""

    policy = _one(_render(_ENABLED_SETTINGS), "NetworkPolicy", f"{_FULLNAME}-vllm")
    prometheus_rules = [
        rule
        for rule in policy["spec"]["ingress"]
        if {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}}}
        in rule["from"]
    ]

    assert len(prometheus_rules) == 1
    assert prometheus_rules[0]["ports"] == [{"port": "http", "protocol": "TCP"}]
    assert all(
        "namespaceSelector" not in peer
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
    )
    assert ".Values.modelServing.networkPolicy.extraIngress" in _NETWORK_POLICY_TEMPLATE.read_text(
        encoding="utf-8"
    )


def test_the_gateway_exports_metrics_and_restarts_when_its_config_changes() -> None:
    """Turn the callback on with monitoring, through a change the pods actually see."""

    on = _render(_ENABLED_SETTINGS)
    off = _render(())

    config = yaml.safe_load(_gateway_config(on))
    assert config["litellm_settings"]["callbacks"] == ["prometheus"]
    assert config["litellm_settings"]["drop_params"] is False
    checksums = {
        state: _one(rendered, "Deployment", f"{_FULLNAME}-litellm")["spec"]["template"]["metadata"][
            "annotations"
        ]["checksum/config"]
        for state, rendered in (("on", on), ("off", off))
    }
    assert checksums["on"] != checksums["off"]


def test_the_database_exposes_its_queues_through_a_kept_read_only_role() -> None:
    """Declare the exporter, the role Grafana reads as, and a password that survives upgrades."""

    rendered = _render(_ENABLED_SETTINGS)
    values = _load_yaml(_VALUES)
    role = values["monitoring"]["database"]["role"]
    secret_name = values["monitoring"]["database"]["secretName"]

    cluster = _one(rendered, "Cluster", values["database"]["cloudNativePG"]["clusterName"])
    assert cluster["spec"]["monitoring"]["enablePodMonitor"] is True
    assert cluster["spec"]["monitoring"]["customQueriesConfigMap"] == [
        {"name": f"{_FULLNAME}-postgres-queries", "key": "queries.yaml"}
    ]
    (managed_role,) = cluster["spec"]["managed"]["roles"]
    assert managed_role["name"] == role
    assert managed_role["login"] is True
    assert managed_role["inRoles"] == ["pg_read_all_data"]
    assert managed_role["passwordSecret"] == {"name": secret_name}
    assert managed_role["ensure"] == "present"

    credential = _one(rendered, "Secret", secret_name)
    assert credential["type"] == "kubernetes.io/basic-auth"
    assert credential["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"
    assert credential["stringData"]["username"] == role
    password = credential["stringData"]["password"]
    assert len(password) >= 32

    datasource_secret = _one(rendered, "Secret", f"{_FULLNAME}-grafana-datasource")
    assert datasource_secret["metadata"]["labels"]["grafana_datasource"] == "1"
    (datasource,) = yaml.safe_load(datasource_secret["stringData"]["flakegraph-postgres.yaml"])[
        "datasources"
    ]
    assert datasource["uid"] == "flakegraph-postgres"
    assert datasource["type"] == "grafana-postgresql-datasource"
    assert datasource["url"] == "flakegraph-postgres-rw:5432"
    assert datasource["database"] == values["database"]["cloudNativePG"]["database"]
    assert datasource["user"] == role
    assert datasource["secureJsonData"]["password"] == password
    assert datasource["jsonData"]["sslmode"] == "require"
    assert datasource["editable"] is False
    # The credential belongs in Secrets and nowhere a ConfigMap reader can see.
    for doc in rendered:
        if doc["kind"] == "ConfigMap":
            assert password not in json.dumps(doc)


def test_grafana_shares_the_fleets_front_door() -> None:
    """Publish Grafana behind the same gate, and let only the edge talk to it."""

    rendered = _render(_ENABLED_SETTINGS)
    values = _load_yaml(_VALUES)
    grafana_host = f"{values['monitoring']['grafana']['host']}.example.test"
    middleware = "traefik.ingress.kubernetes.io/router.middlewares"

    main = _one(rendered, "Ingress", _FULLNAME)
    grafana = _one(rendered, "Ingress", f"{_FULLNAME}-grafana")
    assert (
        grafana["metadata"]["annotations"][middleware]
        == main["metadata"]["annotations"][middleware]
    )
    assert grafana["spec"]["tls"] == [{"secretName": "wildcard-tls", "hosts": [grafana_host]}]
    (rule,) = grafana["spec"]["rules"]
    assert rule["host"] == grafana_host
    (path,) = rule["http"]["paths"]
    assert path["backend"]["service"] == {
        "name": f"{values['monitoring']['release']}-grafana",
        "port": {"number": values["monitoring"]["grafana"]["servicePort"]},
    }

    auth = _one(rendered, "Ingress", f"{_FULLNAME}-auth")
    oauth_hosts = {
        rule["host"]
        for rule in auth["spec"]["rules"]
        if any(path["path"] == "/oauth2/" for path in rule["http"]["paths"])
    }
    assert grafana_host in oauth_hosts
    assert grafana_host in auth["spec"]["tls"][0]["hosts"]

    policy = _one(rendered, "NetworkPolicy", f"{_FULLNAME}-grafana")
    assert policy["spec"]["podSelector"] == {"matchLabels": {"app.kubernetes.io/name": "grafana"}}
    (ingress_rule,) = policy["spec"]["ingress"]
    assert ingress_rule["from"] == [
        *values["controlPlane"]["networkPolicy"]["from"],
        {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}}},
    ]


def test_alerts_cover_each_plane_and_take_every_threshold_from_values() -> None:
    """Keep every alert explained, actionable, and tunable without editing PromQL."""

    rendered = _render(_ENABLED_SETTINGS)
    values = _load_yaml(_VALUES)
    template = _RULES_TEMPLATE.read_text(encoding="utf-8")

    prometheus_rule = _one(rendered, "PrometheusRule", _FULLNAME)
    groups = {group["name"]: group["rules"] for group in prometheus_rule["spec"]["groups"]}
    assert set(groups) == {"flakegraph.serving", "flakegraph.documents", "flakegraph.pipeline"}
    alerts = {rule["alert"]: rule for group in groups.values() for rule in group}
    expected = {
        "FlakeGraphEnginePreempted",
        "FlakeGraphEngineDown",
        "FlakeGraphKvCacheSaturated",
        "FlakeGraphTimeToFirstTokenSlow",
        "FlakeGraphGatewayDown",
        "FlakeGraphOcrShimDown",
        "FlakeGraphOcrQueueStalled",
        "FlakeGraphOcrUpstreamFailing",
        "FlakeGraphTasksFailing",
        "FlakeGraphTasksNearAttemptLimit",
        "FlakeGraphTaskLeasesExpired",
        "FlakeGraphWorkerDemandUnmet",
    }
    assert expected <= set(alerts)
    for rule in (rule for group in groups.values() for rule in group):
        assert rule["labels"]["severity"] in {"critical", "warning"}
        assert rule["annotations"]["summary"]
        assert rule["annotations"]["description"]
        assert rule["annotations"]["runbook"] == "docs/kubernetes-fleet.md#verifying-a-fleet"
        # Helm must hand Prometheus's own templates through untouched.
        assert "`" not in json.dumps(rule)

    assert alerts["FlakeGraphEnginePreempted"]["labels"]["severity"] == "critical"
    assert alerts["FlakeGraphEnginePreempted"]["expr"].startswith(
        "increase(vllm:num_preemptions_total"
    )
    assert alerts["FlakeGraphEngineDown"]["expr"] == (
        f'up{{job="{_FULLNAME}-vllm-headless", namespace="{_NAMESPACE}"}} == 0'
    )
    assert alerts["FlakeGraphGatewayDown"]["expr"].startswith(f'up{{job="{_FULLNAME}-litellm"')
    assert alerts["FlakeGraphOcrShimDown"]["expr"].startswith(f'up{{job="{_FULLNAME}-ocr"')
    assert "{{ $labels.engine }}" in alerts["FlakeGraphEnginePreempted"]["annotations"]["summary"]
    assert "{{ $value" in alerts["FlakeGraphKvCacheSaturated"]["annotations"]["description"]

    thresholds = values["monitoring"]["rules"]["thresholds"]
    for key in thresholds:
        assert f"$t.{key}" in template, f"threshold {key} is declared but no rule uses it"
    assert alerts["FlakeGraphKvCacheSaturated"]["for"] == thresholds["kvCacheUsageFor"]
    assert alerts["FlakeGraphKvCacheSaturated"]["expr"].endswith(f"> {thresholds['kvCacheUsage']}")

    # One demand rule per pool, comparing what KEDA would ask for - tasks per
    # replica, capped at the ceiling - rather than raw demand.
    demand_rules = [
        rule
        for group in groups.values()
        for rule in group
        if rule["alert"] == "FlakeGraphWorkerDemandUnmet"
    ]
    assert {rule["labels"]["pool"] for rule in demand_rules} == set(values["workers"])
    for rule in demand_rules:
        pool = values["workers"][rule["labels"]["pool"]]
        assert f"{pool['autoscaling']['maxReplicas']}\n" in rule["expr"] + "\n"
        assert f'deployment="{_FULLNAME}-{rule["labels"]["pool"]}"' in rule["expr"]
        assert rule["for"] == thresholds["workerDemandUnmetFor"]


def test_postgres_queries_are_well_formed_and_name_only_real_columns() -> None:
    """Keep the exporter's SQL runnable against the schema the adapter actually creates.

    A query naming a column the DDL does not have fails on every scrape, and
    the failure surfaces as a missing series rather than as an error anyone
    reads. So the columns are checked against the adapter's own statements.
    """

    rendered = _render(_ENABLED_SETTINGS)
    values = _load_yaml(_VALUES)
    queries = yaml.safe_load(
        _one(rendered, "ConfigMap", f"{_FULLNAME}-postgres-queries")["data"]["queries.yaml"]
    )
    columns = _schema_columns()
    stages = {stage for pool in values["workers"].values() for stage in pool["stages"]}

    expected = {
        "flakegraph_tasks",
        "flakegraph_tasks_failed_recent",
        "flakegraph_tasks_near_attempt_limit",
        "flakegraph_tasks_lease_expired",
        "flakegraph_tasks_oldest_queued",
        "flakegraph_worker_demand",
        "flakegraph_runs",
        "flakegraph_publications",
    }
    assert set(queries) == expected
    for name, query in queries.items():
        assert query["target_databases"] == [values["database"]["cloudNativePG"]["database"]], name
        assert query["primary"] is True, name
        for metric in query["metrics"]:
            ((column, spec),) = metric.items()
            assert spec["usage"] in {"LABEL", "GAUGE"}, (name, column)
            assert spec["description"], (name, column)
        _assert_query_names_real_columns(name, query["query"], columns)

    # The band split has to be the one the demand view and KEDA scale on.
    adapter = _POSTGRES_ADAPTER.read_text(encoding="utf-8")
    assert "WHEN demand.priority >= 1000 THEN 'interactive'" in adapter
    for name in ("flakegraph_tasks", "flakegraph_tasks_oldest_queued"):
        assert "priority >= 1000 THEN 'interactive'" in queries[name]["query"]
    # Every stage the DDL admits reports a zero rather than vanishing.
    for name in (
        "flakegraph_tasks_failed_recent",
        "flakegraph_tasks_near_attempt_limit",
        "flakegraph_tasks_lease_expired",
    ):
        assert stages == set(re.findall(r"'([a-z_]+)'", queries[name]["query"].split("]")[0])), name


_SQL_WORDS = {
    "and",
    "array",
    "as",
    "by",
    "case",
    "coalesce",
    "cross",
    "count",
    "else",
    "end",
    "epoch",
    "exists",
    "extract",
    "float",
    "from",
    "group",
    "hour",
    "in",
    "interval",
    "join",
    "left",
    "min",
    "not",
    "now",
    "on",
    "or",
    "select",
    "then",
    "unnest",
    "when",
    "where",
}


def _assert_query_names_real_columns(name: str, sql: str, columns: dict[str, set[str]]) -> None:
    """Fail on the first identifier that is neither SQL, an alias, nor a real column."""

    without_literals = re.sub(r"'[^']*'", "''", sql)
    tables = set(re.findall(r"\b(?:FROM|JOIN)\s+(flakegraph_[a-z_]+)", without_literals))
    assert tables, name
    aliases = set(re.findall(r"\bAS\s+([a-z_]+)", without_literals))
    aliases |= {
        match for match in re.findall(r"\b([a-z_]+)\(", without_literals) if match not in _SQL_WORDS
    }
    aliases |= set(re.findall(r"\bAS\s+[a-z_]+\(([a-z_]+)\)", without_literals))
    known = set().union(*(columns[table] for table in tables))
    for word in set(re.findall(r"\b[a-z_]+\b", without_literals.lower())):
        if word in _SQL_WORDS or word in tables or word in aliases:
            continue
        assert word in known, f"{name} names {word!r}, which is not a column of {sorted(tables)}"


@cache
def _schema_columns() -> dict[str, set[str]]:
    """Read table and view columns out of the adapter's own DDL strings."""

    adapter = _POSTGRES_ADAPTER.read_text(encoding="utf-8")
    columns: dict[str, set[str]] = {}
    for match in re.finditer(
        r"CREATE TABLE IF NOT EXISTS (flakegraph_[a-z_]+) \((.*?)\n    \)", adapter, re.DOTALL
    ):
        table, body = match.groups()
        columns[table] = set(
            re.findall(
                r"^\s*([a-z_]+)\s+(?:TEXT|INTEGER|TIMESTAMPTZ|JSONB|BIGSERIAL|BIGINT)",
                body,
                re.MULTILINE,
            )
        )
    for match in re.finditer(r"ADD COLUMN IF NOT EXISTS ([a-z_]+)", adapter):
        columns["flakegraph_task"].add(match.group(1))
    view = re.search(
        r"CREATE VIEW (flakegraph_[a-z_]+) AS\n    SELECT (.*?)\n    FROM", adapter, re.DOTALL
    )
    assert view is not None
    # Output columns are either aliased or a bare `demand.<column>,` entry.
    columns[view.group(1)] = set(re.findall(r"\bAS ([a-z_]+)", view.group(2))) | set(
        re.findall(r"\bdemand\.([a-z_]+),", view.group(2))
    )
    assert {
        "stage",
        "status",
        "priority",
        "attempts",
        "max_attempts",
        "lease_expires_at",
    } <= columns["flakegraph_task"]
    assert columns["flakegraph_worker_demand"] == {"stage", "priority_band", "desired_workers"}
    return columns


def _gateway_config(rendered: list[dict[str, Any]]) -> str:
    return _one(rendered, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]


def _one(rendered: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Return exactly one rendered object by kind and name."""

    matches = [doc for doc in rendered if doc["kind"] == kind and doc["metadata"]["name"] == name]
    assert len(matches) == 1, f"expected one {kind}/{name}, found {len(matches)}"
    return matches[0]


def _helm() -> str:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed; the rendered-chart contracts need it")
    return helm


@cache
def _render(settings: tuple[str, ...]) -> list[dict[str, Any]]:
    """Render the chart offline, with the operator CRDs declared present."""

    command = [
        _helm(),
        "template",
        _RELEASE,
        str(_CHART),
        "--namespace",
        _NAMESPACE,
        "--api-versions",
        _API_VERSIONS,
    ]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one repository-owned YAML mapping for contract assertions."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
