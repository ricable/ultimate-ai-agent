#!/bin/bash
# Network configuration for Thunderbolt ring + 10GbE setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Node configuration
declare -A NODES=(
    ["mac-node-1"]="10.0.1.10"
    ["mac-node-2"]="10.0.1.11" 
    ["mac-node-3"]="10.0.1.12"
)

# Configure jumbo frames for 10GbE
configure_jumbo_frames() {
    local interface=$1
    log "Configuring jumbo frames on $interface"
    
    # Check if interface exists
    if ! ifconfig "$interface" &> /dev/null; then
        warn "Interface $interface not found, skipping jumbo frame configuration"
        return 0
    fi
    
    # Set MTU to 9000 for jumbo frames
    log "Setting MTU to 9000 on $interface"
    sudo ifconfig "$interface" mtu 9000
    
    # Make persistent across reboots
    log "Making MTU configuration persistent"
    sudo networksetup -setMTU "$interface" 9000
    
    # Verify configuration
    local mtu=$(ifconfig "$interface" | grep mtu | awk '{print $6}')
    log "Current MTU on $interface: $mtu"
    
    if [ "$mtu" = "9000" ]; then
        log "✓ Jumbo frames configured successfully on $interface"
    else
        warn "Jumbo frames configuration may have failed on $interface"
    fi
}

# Setup Thunderbolt bridge for ring topology
setup_thunderbolt_bridge() {
    log "Setting up Thunderbolt bridge"
    
    # Create bridge for Thunderbolt interfaces
    if ! ifconfig bridge0 &> /dev/null; then
        log "Creating bridge0 interface"
        sudo ifconfig bridge0 create 2>/dev/null || true
    else
        log "Bridge0 already exists"
    fi
    
    # Find Thunderbolt interfaces (typically en5, en6, etc.)
    log "Discovering Thunderbolt interfaces..."
    TB_INTERFACES=$(networksetup -listallhardwareports | grep -A1 "Thunderbolt" | grep "Device:" | awk '{print $2}' | tr '\n' ' ')
    
    if [ -z "$TB_INTERFACES" ]; then
        warn "No Thunderbolt interfaces found"
        return 0
    fi
    
    log "Found Thunderbolt interfaces: $TB_INTERFACES"
    
    # Add Thunderbolt interfaces to bridge
    for iface in $TB_INTERFACES; do
        if ifconfig "$iface" &> /dev/null; then
            log "Adding $iface to bridge0"
            sudo ifconfig bridge0 addm "$iface" 2>/dev/null || warn "Failed to add $iface to bridge"
        fi
    done
    
    # Bring up the bridge
    log "Bringing up bridge0"
    sudo ifconfig bridge0 up
    
    # Verify bridge configuration
    if ifconfig bridge0 | grep -q "UP"; then
        log "✓ Thunderbolt bridge configured successfully"
    else
        warn "Thunderbolt bridge configuration may have failed"
    fi
}

# Configure firewall for MLX and Exo
configure_firewall() {
    log "Configuring firewall rules"
    
    # Check if pfctl is available
    if ! command -v pfctl &> /dev/null; then
        warn "pfctl not available, skipping firewall configuration"
        return 0
    fi
    
    # Enable packet filter
    log "Enabling packet filter"
    sudo pfctl -e 2>/dev/null || warn "Packet filter may already be enabled"
    
    # Create temporary pf rules file
    local rules_file="/tmp/mlx_exo_rules.conf"
    log "Creating firewall rules file: $rules_file"
    
    cat > "$rules_file" << 'EOF'
# MLX distributed ports
pass in proto tcp to any port 40000:40100
pass in proto udp to any port 40000:40100

# Exo P2P ports  
pass in proto tcp to any port 52415
pass in proto udp to any port 52415

# SSH for remote coordination
pass in proto tcp to any port 22

# mDNS for discovery
pass in proto udp to any port 5353

# HTTP/HTTPS for API
pass in proto tcp to any port 80
pass in proto tcp to any port 443
pass in proto tcp to any port 8000

# Ray framework ports
pass in proto tcp to any port 6379
pass in proto tcp to any port 8265
pass in proto tcp to any port 10001

# FastAPI ports
pass in proto tcp to any port 8000:8010
EOF
    
    # Backup existing pf.conf
    if [ -f /etc/pf.conf ]; then
        log "Backing up existing pf.conf"
        sudo cp /etc/pf.conf /etc/pf.conf.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    # Add rules to pf.conf (avoid duplicates)
    log "Adding rules to pf.conf"
    if ! grep -q "MLX distributed ports" /etc/pf.conf 2>/dev/null; then
        echo "" | sudo tee -a /etc/pf.conf > /dev/null
        echo "# MLX-Exo cluster rules" | sudo tee -a /etc/pf.conf > /dev/null
        sudo cat "$rules_file" >> /etc/pf.conf
        log "Firewall rules added to pf.conf"
    else
        log "MLX-Exo firewall rules already exist in pf.conf"
    fi
    
    # Reload firewall rules
    log "Reloading firewall rules"
    sudo pfctl -f /etc/pf.conf || warn "Failed to reload firewall rules"
    
    # Clean up temporary file
    rm -f "$rules_file"
    
    log "✓ Firewall configured successfully"
}

# Test network connectivity
test_connectivity() {
    log "Testing network connectivity"
    
    local failed=0
    local reachable=0
    
    for node in "${!NODES[@]}"; do
        local ip="${NODES[$node]}"
        log "Testing connectivity to $node at $ip"
        
        if ping -c 2 -W 2000 "$ip" > /dev/null 2>&1; then
            log "✓ $node at $ip reachable"
            ((reachable++))
        else
            warn "✗ Cannot reach $node at $ip"
            ((failed++))
        fi
    done
    
    log "Connectivity test results: $reachable reachable, $failed unreachable"
    
    if [ $failed -eq 0 ]; then
        log "✓ All nodes reachable"
    else
        warn "⚠ $failed nodes unreachable - this may be expected if nodes are not yet configured"
    fi
}

# Configure SSH for passwordless access
setup_ssh_keys() {
    log "Setting up SSH keys for passwordless access"
    
    # Generate SSH key if it doesn't exist
    if [ ! -f ~/.ssh/mlx_cluster ]; then
        log "Generating SSH key for cluster communication"
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/mlx_cluster -N "" -C "mlx-cluster-$(whoami)"
    else
        log "SSH key already exists"
    fi
    
    # Configure SSH client
    log "Configuring SSH client settings"
    if ! grep -q "Host mac-node-*" ~/.ssh/config 2>/dev/null; then
        cat >> ~/.ssh/config << 'EOF'

# MLX Cluster configuration
Host mac-node-*
    User mlx
    IdentityFile ~/.ssh/mlx_cluster
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
        log "SSH client configuration added"
    else
        log "SSH client configuration already exists"
    fi
    
    log "✓ SSH keys configured successfully"
    log "Note: You'll need to copy the public key to other nodes manually:"
    log "Public key location: ~/.ssh/mlx_cluster.pub"
}

# Detect and configure primary network interface
detect_primary_interface() {
    log "Detecting primary network interface"
    
    # Try to get the interface used for default route
    local interface=$(route get default 2>/dev/null | awk '/interface:/ {print $2}')
    
    if [ -z "$interface" ]; then
        # Fallback: find first active ethernet interface
        interface=$(ifconfig | grep -E "^en[0-9]:" | head -1 | cut -d: -f1)
    fi
    
    if [ -z "$interface" ]; then
        warn "Could not detect primary network interface"
        return 1
    fi
    
    log "Primary network interface: $interface"
    echo "$interface"
}

# Create network diagnostic script
create_diagnostic_script() {
    log "Creating network diagnostic script"
    
    cat > ~/network_diagnostics.sh << 'EOF'
#!/bin/bash
# Network diagnostics for MLX cluster

echo "=== Network Interface Status ==="
ifconfig | grep -E "(^en[0-9]|^bridge[0-9]|inet |mtu)" | grep -A2 -B1 "inet"

echo -e "\n=== Routing Table ==="
netstat -rn | head -10

echo -e "\n=== Active Network Connections ==="
netstat -an | grep -E "(52415|40000|8000)" | head -10

echo -e "\n=== Firewall Status ==="
sudo pfctl -sr 2>/dev/null | grep -E "(52415|40000|8000)" || echo "No specific rules found"

echo -e "\n=== Thunderbolt Interfaces ==="
networksetup -listallhardwareports | grep -A1 "Thunderbolt" || echo "No Thunderbolt interfaces found"

echo -e "\n=== Bridge Status ==="
ifconfig bridge0 2>/dev/null || echo "No bridge0 interface found"

echo -e "\n=== Node Connectivity Test ==="
for ip in 10.0.1.10 10.0.1.11 10.0.1.12; do
    if ping -c 1 -W 1000 $ip > /dev/null 2>&1; then
        echo "✓ $ip reachable"
    else
        echo "✗ $ip unreachable"
    fi
done
EOF
    
    chmod +x ~/network_diagnostics.sh
    log "Network diagnostic script created at ~/network_diagnostics.sh"
}

# Main execution
main() {
    log "Starting network configuration for MLX-Exo cluster"
    
    # Check if running as root for certain operations
    if [ "$EUID" -eq 0 ]; then
        warn "Running as root - some operations may behave differently"
    fi
    
    # Detect primary interface
    PRIMARY_INTERFACE=$(detect_primary_interface)
    if [ $? -eq 0 ] && [ -n "$PRIMARY_INTERFACE" ]; then
        configure_jumbo_frames "$PRIMARY_INTERFACE"
    fi
    
    setup_thunderbolt_bridge
    configure_firewall
    setup_ssh_keys
    test_connectivity
    create_diagnostic_script
    
    log "Network configuration complete!"
    log ""
    log "Next steps:"
    log "1. Copy SSH public key (~/.ssh/mlx_cluster.pub) to other nodes"
    log "2. Configure static IP addresses on each node"
    log "3. Run network diagnostics: ~/network_diagnostics.sh"
    log "4. Test node-to-node connectivity"
}

# Handle command line arguments
case "${1:-}" in
    --test-only)
        log "Running connectivity test only"
        test_connectivity
        ;;
    --diagnostics)
        log "Creating diagnostic script only"
        create_diagnostic_script
        ;;
    *)
        main "$@"
        ;;
esac