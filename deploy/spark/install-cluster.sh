#!/usr/bin/env bash
# Install the workload layer FlakeGraph expects on a freshly bootstrapped Spark.
#
# bootstrap-node.sh leaves a running k3s with a GPU-capable container runtime and
# nothing else. This script adds the pieces the FlakeGraph chart depends on but
# does not own: GPU scheduling, queue-driven autoscaling, PostgreSQL for task
# leases, and S3-compatible storage for artifacts.
#
# It runs unprivileged against the kubeconfig k3s wrote, because none of it needs
# root — keeping the privileged surface confined to bootstrap-node.sh is the
# point of the split.
#
#   ./install-cluster.sh
#
# Everything is installed with `helm upgrade --install`, so re-running converges
# rather than duplicating. Chart versions are pinned: an unpinned fleet drifts
# apart one node at a time and the differences only surface under load.
set -euo pipefail

namespace="${FLAKEGRAPH_NAMESPACE:-flakegraph}"
helm_version="${FLAKEGRAPH_HELM_VERSION:-v3.19.0}"
bin_dir="${FLAKEGRAPH_BIN_DIR:-$HOME/.local/bin}"

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="$bin_dir:$PATH"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }

if ! kubectl get nodes >/dev/null 2>&1; then
  echo "cannot reach the cluster with KUBECONFIG=$KUBECONFIG" >&2
  echo "run bootstrap-node.sh --role server on this node first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
step "Helm"

if ! command -v helm >/dev/null 2>&1; then
  arch="$(uname -m)"; [[ "$arch" == "aarch64" ]] && arch=arm64 || arch=amd64
  mkdir -p "$bin_dir"
  tmp="$(mktemp -d)"
  # get.helm.sh is a plain object store, unlike the GitHub release host this
  # network blocks, so helm can be fetched on the node itself.
  curl -fsSL "https://get.helm.sh/helm-${helm_version}-linux-${arch}.tar.gz" \
    | tar -xz -C "$tmp"
  install -m 0755 "$tmp/linux-${arch}/helm" "$bin_dir/helm"
  rm -rf "$tmp"
fi
ok "$(helm version --short)"

# ---------------------------------------------------------------------------
step "Chart repositories"

add_repo() {
  helm repo add "$1" "$2" >/dev/null 2>&1 || true
}
add_repo nvdp https://nvidia.github.io/k8s-device-plugin
add_repo kedacore https://kedacore.github.io/charts
add_repo cnpg https://cloudnative-pg.github.io/charts
add_repo minio https://charts.min.io/
helm repo update >/dev/null
ok "nvdp, kedacore, cnpg, minio"

# ---------------------------------------------------------------------------
step "GPU scheduling"

# Without the device plugin the GB10 is invisible to the scheduler: pods that
# request nvidia.com/gpu stay Pending forever, and pods that do not request it
# run on the CPU while appearing healthy. The runtimeClass comes from k3s, which
# generates it when it detects the NVIDIA container runtime at startup.
helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --version 0.17.4 \
  --set runtimeClassName=nvidia \
  --set-string nodeSelector."nvidia\.com/gpu\.present"=true \
  --wait --timeout 5m >/dev/null
ok "nvidia device plugin installed"

# ---------------------------------------------------------------------------
step "Queue-driven autoscaling"

# FlakeGraph scales workers from the depth of its PostgreSQL task queue rather
# than from CPU, so KEDA is a hard dependency of the chart's autoscaling paths.
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace \
  --version 2.17.2 \
  --wait --timeout 10m >/dev/null
ok "keda installed"

# ---------------------------------------------------------------------------
step "PostgreSQL operator"

# The FlakeGraph chart can create its own CloudNativePG Cluster, but the
# operator that reconciles it is cluster-scoped and therefore installed here.
#
# Pulled from the OCI registry rather than the https repository: the index on
# cloudnative-pg.github.io resolves chart tarballs to GitHub's release-asset
# host, which this network intercepts with an untrusted certificate. ghcr.io
# serves the identical chart and is reachable.
helm upgrade --install cnpg oci://ghcr.io/cloudnative-pg/charts/cloudnative-pg \
  --namespace cnpg-system --create-namespace \
  --version 0.26.0 \
  --wait --timeout 10m >/dev/null
ok "cloudnative-pg operator installed"

# ---------------------------------------------------------------------------
step "Object storage"

# Payload bytes and Parquet graph tables move through S3, not through the
# database. A single-node MinIO is right for one Spark and is the piece to
# replace with a managed endpoint when the fleet outgrows this machine.
kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

if ! kubectl -n "$namespace" get secret flakegraph-artifacts >/dev/null 2>&1; then
  # Generated once and stored only in the cluster. A checked-in default would be
  # the same password on every Spark LEO ever racks.
  kubectl -n "$namespace" create secret generic flakegraph-artifacts \
    --from-literal=rootUser=flakegraph \
    --from-literal=rootPassword="$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 32)" >/dev/null
  ok "generated object storage credentials"
fi
root_user="$(kubectl -n "$namespace" get secret flakegraph-artifacts -o jsonpath='{.data.rootUser}' | base64 -d)"
root_password="$(kubectl -n "$namespace" get secret flakegraph-artifacts -o jsonpath='{.data.rootPassword}' | base64 -d)"

helm upgrade --install minio minio/minio \
  --namespace "$namespace" \
  --version 5.4.0 \
  --set mode=standalone \
  --set replicas=1 \
  --set persistence.size=200Gi \
  --set resources.requests.memory=2Gi \
  --set-string rootUser="$root_user" \
  --set-string rootPassword="$root_password" \
  --set buckets[0].name=flakegraph-artifacts \
  --set buckets[0].policy=none \
  --set buckets[0].purge=false \
  --wait --timeout 10m >/dev/null
ok "minio installed with bucket flakegraph-artifacts"

# ---------------------------------------------------------------------------
step "Done"

cat <<SUMMARY
    Namespace:      $namespace
    Object storage: http://minio.$namespace.svc.cluster.local:9000
    Credentials:    secret/flakegraph-artifacts in $namespace

    Install FlakeGraph itself with the chart in deploy/helm/flakegraph, using a
    values file derived from deploy/examples/k3s-spark-values.yaml.
SUMMARY
