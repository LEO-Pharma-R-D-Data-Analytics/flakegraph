#!/usr/bin/env bash
# Prepare an NVIDIA DGX Spark to run FlakeGraph as a Kubernetes node.
#
# Corporate Sparks arrive with a CIS-hardened Ubuntu image that blacklists the
# overlay filesystem. Containers cannot run without it: Docker's overlay2
# graphdriver and the containerd that k3s bundles both mount overlay for every
# layer, so a hardened Spark has a working GPU, a working driver, and no way to
# start a container. This script applies that one exception, then brings up the
# container runtime and the cluster on top of it.
#
# Every node in the fleet runs this same script. The first Spark takes --role
# server, which starts an etcd-backed control plane that further Sparks join
# rather than a single-node cluster that would have to be rebuilt to grow.
#
# Run it with sudo, from the node itself:
#   sudo ./bootstrap-node.sh --role server
#   sudo ./bootstrap-node.sh --role agent --server https://10.217.20.10:6443 --token <token>
#
# It is idempotent: re-running reports what is already in place and changes
# nothing else, so it doubles as a way to verify a node still matches the fleet.
set -euo pipefail

role=""
server_url=""
join_token=""
node_class="${FLAKEGRAPH_NODE_CLASS:-nvidia-spark}"
backup_root="${FLAKEGRAPH_BACKUP_ROOT:-/etc/flakegraph/backups}"
k3s_channel="${FLAKEGRAPH_K3S_CHANNEL:-stable}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) role="${2:-}"; shift 2 ;;
    --server) server_url="${2:-}"; shift 2 ;;
    --token) join_token="${2:-}"; shift 2 ;;
    --node-class) node_class="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$role" in
  server|agent) ;;
  "") echo "--role is required (server for the first Spark, agent for the rest)" >&2; exit 2 ;;
  *) echo "--role must be 'server' or 'agent'" >&2; exit 2 ;;
esac

if [[ "$role" == "agent" && ( -z "$server_url" || -z "$join_token" ) ]]; then
  echo "--role agent requires --server and --token from an existing server node" >&2
  echo "read the token there with: sudo cat /var/lib/rancher/k3s/server/node-token" >&2
  exit 2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "this script configures kernel modules and system services; run it with sudo" >&2
  exit 1
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }

# ---------------------------------------------------------------------------
step "Preflight"

arch="$(uname -m)"
[[ "$arch" == "aarch64" ]] || ok "warning: expected aarch64 (GB10), found $arch"
ok "kernel $(uname -r) on $(. /etc/os-release && echo "$PRETTY_NAME")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: install the NVIDIA driver before joining the fleet" >&2
  exit 1
fi
ok "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

mkdir -p "$backup_root"

# ---------------------------------------------------------------------------
step "Overlay filesystem exception"

# CIS hardening disables a set of uncommon filesystems to shrink the kernel's
# attack surface. overlay is in that set, but unlike cramfs or hfsplus it is not
# uncommon on this machine: it is the mechanism every container image uses. The
# blacklist is therefore lifted here rather than worked around, because the
# alternatives are worse — the vfs graphdriver copies every layer in full and
# would turn a 12 GB CUDA image into tens of gigabytes per container, and
# fuse-overlayfs moves the same code into userspace without removing it.
#
# The original file is kept, with its directives commented rather than deleted,
# so an auditor reading this node sees the control and the reason it was lifted
# in the same place.
overlay_conf=/etc/modprobe.d/cis_overlay.conf
if [[ -f "$overlay_conf" ]] && grep -qE '^\s*(install\s+overlay|blacklist\s+overlay)' "$overlay_conf"; then
  cp -a "$overlay_conf" "$backup_root/cis_overlay.conf.$(date +%Y%m%d%H%M%S)"
  {
    echo "# Modified by FlakeGraph bootstrap-node.sh."
    echo "#"
    echo "# The CIS directives below are retained for reference and deliberately"
    echo "# inert. This host is a container node: overlay is required by Docker's"
    echo "# overlay2 graphdriver and by the containerd that k3s embeds, and no"
    echo "# container can start while the module is blocked. Restore the file from"
    echo "# $backup_root to put the control back."
    echo "#"
    sed 's/^\(\s*\(install\|blacklist\)\s\+overlay\b\)/# \1/' "$overlay_conf"
  } > "$overlay_conf.new"
  mv "$overlay_conf.new" "$overlay_conf"
  chmod 644 "$overlay_conf"
  ok "lifted the overlay blacklist (original backed up under $backup_root)"
else
  ok "overlay blacklist already lifted"
fi

# Persist the module across reboots. Docker starts before anything would mount
# an overlay filesystem on demand, so relying on autoload leaves the daemon
# failing on every boot even once the blacklist is gone.
echo "overlay" > /etc/modules-load.d/flakegraph-overlay.conf
modprobe overlay
grep -q '^nodev\?\s*overlay$' /proc/filesystems || grep -q overlay /proc/filesystems
ok "overlay module loaded and set to load at boot"

# ---------------------------------------------------------------------------
step "Docker"

if ! command -v dockerd >/dev/null 2>&1; then
  echo "dockerd is not installed; install docker-ce before running this script" >&2
  exit 1
fi

systemctl reset-failed docker.service docker.socket 2>/dev/null || true
systemctl enable --now docker >/dev/null 2>&1 || systemctl start docker
sleep 2
if ! docker info >/dev/null 2>&1; then
  echo "docker failed to start; see: journalctl -u docker -n 50" >&2
  exit 1
fi
ok "docker $(docker version --format '{{.Server.Version}}') using the $(docker info --format '{{.Driver}}') driver"

# The NVIDIA container runtime is what lets a build or an ad-hoc container see
# the GB10. The toolkit ships with the DGX image; this only wires it into the
# daemon's runtime list, and leaves runc as the default so ordinary builds are
# unaffected.
if command -v nvidia-ctk >/dev/null 2>&1; then
  if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    nvidia-ctk runtime configure --runtime=docker >/dev/null
    systemctl restart docker
    sleep 2
    ok "registered the nvidia runtime with docker"
  else
    ok "nvidia runtime already registered with docker"
  fi
fi

# ---------------------------------------------------------------------------
step "Kubernetes (k3s)"

if systemctl is-active --quiet k3s || systemctl is-active --quiet k3s-agent; then
  ok "k3s already running; leaving the existing installation alone"
else
  primary_ip="$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)"
  export INSTALL_K3S_CHANNEL="$k3s_channel"

  if [[ "$role" == "server" ]]; then
    # --cluster-init starts embedded etcd. A default k3s server uses SQLite and
    # can never gain a second control-plane node, which would mean rebuilding
    # the cluster the first time this fleet grows beyond one Spark.
    #
    # The node's own address is added as a TLS SAN so an operator's kubectl can
    # reach the API over the corporate network rather than only from localhost.
    INSTALL_K3S_EXEC="server --cluster-init \
      --tls-san ${primary_ip} \
      --tls-san $(hostname -f 2>/dev/null || hostname) \
      --write-kubeconfig-mode 0644 \
      --node-label flakegraph.io/node-class=${node_class} \
      --node-label nvidia.com/gpu.present=true" \
      curl -sfL https://get.k3s.io | sh -
    ok "k3s server started with embedded etcd"
  else
    INSTALL_K3S_EXEC="agent \
      --node-label flakegraph.io/node-class=${node_class} \
      --node-label nvidia.com/gpu.present=true" \
      K3S_URL="$server_url" K3S_TOKEN="$join_token" \
      curl -sfL https://get.k3s.io | sh -
    ok "k3s agent joined $server_url"
  fi
fi

if [[ "$role" == "server" ]]; then
  # k3s detects the NVIDIA container runtime at startup and generates a
  # containerd config that exposes it. Confirm rather than assume: a missing
  # runtime here surfaces later as GPU pods that schedule and then run on CPU.
  if grep -q 'nvidia' /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
    ok "k3s containerd exposes the nvidia runtime"
  else
    ok "warning: k3s containerd has no nvidia runtime; GPU pods will not work"
  fi

  # Hand the operator a kubeconfig they own. k3s writes one readable copy under
  # /etc, but tools that rewrite contexts need a file they can modify.
  target_user="${SUDO_USER:-root}"
  target_home="$(getent passwd "$target_user" | cut -d: -f6)"
  if [[ -n "$target_home" && -d "$target_home" ]]; then
    install -d -o "$target_user" -g "$target_user" -m 0700 "$target_home/.kube"
    install -o "$target_user" -g "$target_user" -m 0600 \
      /etc/rancher/k3s/k3s.yaml "$target_home/.kube/config"
    sed -i "s#https://127.0.0.1:6443#https://${primary_ip:-127.0.0.1}:6443#" "$target_home/.kube/config"
    ok "wrote $target_home/.kube/config for $target_user"
  fi
fi

# ---------------------------------------------------------------------------
step "Done"

if [[ "$role" == "server" ]]; then
  cat <<SUMMARY
    Control plane:  https://$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1):6443
    Join a Spark:   sudo ./bootstrap-node.sh --role agent \\
                      --server https://$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1):6443 \\
                      --token \$(sudo cat /var/lib/rancher/k3s/server/node-token)

    The cluster has no GPU device plugin yet. Install the workload layer with
    deploy/spark/install-cluster.sh, which runs unprivileged against this
    kubeconfig.
SUMMARY
else
  echo "    Node joined. Verify from a server node with: kubectl get nodes -o wide"
fi
