#!/bin/bash
# Edge-Native AI SaaS - Kairos Node Setup Script
# Bootstraps a new edge node with immutable OS and K3s

set -e

# Configuration
NETWORK_TOKEN="${NETWORK_TOKEN:-}"
K3S_TOKEN="${K3S_TOKEN:-}"
CONTROL_PLANE_VIP="${CONTROL_PLANE_VIP:-10.0.0.1}"
NODE_ROLE="${NODE_ROLE:-worker}"  # master or worker
ZONE="${ZONE:-edge-zone-1}"

echo "=========================================="
echo "Kairos Edge Node Setup"
echo "=========================================="
echo ""

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    ARCH_NAME="arm64"
    ISO_URL="https://github.com/kairos-io/kairos/releases/latest/download/kairos-standard-aarch64.iso"
elif [[ "$ARCH" == "x86_64" ]]; then
    ARCH_NAME="amd64"
    ISO_URL="https://github.com/kairos-io/kairos/releases/latest/download/kairos-standard-amd64.iso"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "Detected architecture: $ARCH_NAME"
echo "Node role: $NODE_ROLE"
echo "Zone: $ZONE"
echo ""

# Check for required tokens
if [[ -z "$NETWORK_TOKEN" ]]; then
    echo "Warning: NETWORK_TOKEN not set. P2P mesh will not be configured."
    echo "Generate one with: npx claude-flow network token"
fi

if [[ -z "$K3S_TOKEN" ]] && [[ "$NODE_ROLE" == "worker" ]]; then
    echo "Warning: K3S_TOKEN not set. Worker node cannot join cluster."
fi

# Generate cloud-config
echo "Generating cloud-config.yaml..."

cat > /tmp/cloud-config.yaml << EOF
#cloud-config
hostname: "edge-node-\$(cat /etc/machine-id | cut -c1-8)"

users:
  - name: "admin"
    groups:
      - "admin"
      - "docker"
      - "wheel"
    ssh_authorized_keys:
      # Add your SSH keys here
    shell: /bin/bash

k3s:
  enabled: true
  args:
    - --disable=traefik
    - --flannel-backend=wireguard-native
    - --node-label=topology.kubernetes.io/zone=$ZONE
    - --node-label=kairos.io/arch=$ARCH_NAME

p2p:
  enabled: true
  network_token: "$NETWORK_TOKEN"
  dns: true
  vpn:
    enabled: true
    use_dht: true

stages:
  boot:
    - name: "Install SpinKube Shim"
      commands:
        - |
          ARCH=\$(uname -m)
          if [ "\$ARCH" = "aarch64" ]; then
            curl -L https://github.com/spinkube/containerd-shim-spin/releases/latest/download/containerd-shim-spin-v2-linux-aarch64.tar.gz | tar xz -C /usr/local/bin
          else
            curl -L https://github.com/spinkube/containerd-shim-spin/releases/latest/download/containerd-shim-spin-v2-linux-x86_64.tar.gz | tar xz -C /usr/local/bin
          fi
          chmod +x /usr/local/bin/containerd-shim-spin-v2

bundles:
  - targets:
      - run://quay.io/kairos/community-bundles:nvidia_latest
EOF

if [[ "$NODE_ROLE" == "master" ]]; then
    cat >> /tmp/cloud-config.yaml << EOF

k3s:
  args:
    - --cluster-init
    - --write-kubeconfig-mode=644
EOF
else
    cat >> /tmp/cloud-config.yaml << EOF

k3s:
  args:
    - --server=https://$CONTROL_PLANE_VIP:6443
    - --token=$K3S_TOKEN
EOF
fi

echo "Cloud-config generated at /tmp/cloud-config.yaml"
echo ""

# Check if we should download the ISO
if [[ "${DOWNLOAD_ISO:-false}" == "true" ]]; then
    echo "Downloading Kairos ISO for $ARCH_NAME..."
    curl -L -o /tmp/kairos-$ARCH_NAME.iso "$ISO_URL"
    echo "ISO downloaded to /tmp/kairos-$ARCH_NAME.iso"
    echo ""
    echo "Next steps:"
    echo "1. Flash the ISO to a USB drive or SD card"
    echo "2. Copy cloud-config.yaml to the boot media"
    echo "3. Boot the target device from the media"
fi

echo ""
echo "=========================================="
echo "Node Configuration Complete"
echo "=========================================="
echo ""
echo "To install on a new node:"
echo "1. Download Kairos ISO: $ISO_URL"
echo "2. Flash to USB/SD card"
echo "3. Copy /tmp/cloud-config.yaml to boot media"
echo "4. Boot the target device"
echo ""
echo "For Raspberry Pi:"
echo "  Use the ARM64 image and flash with Raspberry Pi Imager"
echo ""
echo "For Mac (as worker with virtualization):"
echo "  Use UTM or Parallels with the ARM64 ISO"
echo ""
