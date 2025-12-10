"""
MLX Distributed Configuration
Handles both MPI and Ring backend setup for Apple Silicon cluster
"""

import os
import socket
import subprocess
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    """Configuration for a single cluster node"""
    name: str
    ip: str
    role: str
    memory_gb: int
    gpu_cores: int
    cpu_cores: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass 
class ClusterConfig:
    """Complete cluster configuration"""
    nodes: List[NodeConfig]
    backend: str  # 'mpi' or 'ring'
    network_interface: str
    use_thunderbolt: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'backend': self.backend,
            'network_interface': self.network_interface,
            'use_thunderbolt': self.use_thunderbolt
        }

class MLXDistributedConfig:
    """Manages MLX distributed configuration for Apple Silicon cluster"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.path.expanduser("~/.mlx_cluster_config.json")
        self.cluster_config = self._load_cluster_config()
        self.current_node = self._detect_current_node()
        logger.info(f"Initialized MLX distributed config for node: {self.current_node.name if self.current_node else 'Unknown'}")
        
    def _load_cluster_config(self) -> ClusterConfig:
        """Load cluster configuration from file or use defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    nodes = [NodeConfig(**node) for node in data['nodes']]
                    return ClusterConfig(
                        nodes=nodes,
                        backend=data.get('backend', 'ring'),
                        network_interface=data.get('network_interface', 'en0'),
                        use_thunderbolt=data.get('use_thunderbolt', True)
                    )
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Default configuration
        nodes = [
            NodeConfig("mac-node-1", "10.0.1.10", "compute", 64, 32, 10),
            NodeConfig("mac-node-2", "10.0.1.11", "compute", 64, 32, 10), 
            NodeConfig("mac-node-3", "10.0.1.12", "compute", 32, 30, 12)
        ]
        
        config = ClusterConfig(
            nodes=nodes,
            backend="ring",  # Default to ring for Thunderbolt optimization
            network_interface="en0",
            use_thunderbolt=True
        )
        
        # Save default config
        self._save_cluster_config(config)
        return config
    
    def _save_cluster_config(self, config: ClusterConfig) -> None:
        """Save cluster configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"Saved cluster configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _detect_current_node(self) -> Optional[NodeConfig]:
        """Detect which node this script is running on"""
        hostname = socket.gethostname().lower()
        local_ips = self._get_local_ips()
        
        logger.debug(f"Hostname: {hostname}")
        logger.debug(f"Local IPs: {local_ips}")
        
        for node in self.cluster_config.nodes:
            # Check by IP address
            if node.ip in local_ips:
                logger.info(f"Node detected by IP: {node.name} ({node.ip})")
                return node
            
            # Check by hostname
            if node.name.lower() in hostname or hostname in node.name.lower():
                logger.info(f"Node detected by hostname: {node.name}")
                return node
        
        logger.warning("Could not detect current node")
        return None
    
    def _get_local_ips(self) -> List[str]:
        """Get all local IP addresses"""
        try:
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            ips = []
            for line in lines:
                if 'inet ' in line and '127.0.0.1' not in line:
                    try:
                        ip = line.split('inet ')[1].split(' ')[0]
                        # Validate IP format
                        socket.inet_aton(ip)
                        ips.append(ip)
                    except (IndexError, socket.error):
                        continue
            return ips
        except Exception as e:
            logger.error(f"Failed to get local IPs: {e}")
            return []
    
    def generate_hostfile(self, output_path: str = "cluster_hostfile.json") -> str:
        """Generate MLX hostfile for distributed launch"""
        hostfile = {
            "hosts": [
                {
                    "host": node.ip,
                    "port": 40000 + i,
                    "rank": i,
                    "name": node.name,
                    "memory_gb": node.memory_gb,
                    "gpu_cores": node.gpu_cores,
                    "role": node.role
                }
                for i, node in enumerate(self.cluster_config.nodes)
            ],
            "backend": self.cluster_config.backend,
            "total_ranks": len(self.cluster_config.nodes),
            "network_interface": self.cluster_config.network_interface,
            "use_thunderbolt": self.cluster_config.use_thunderbolt
        }
        
        output_path = os.path.expanduser(output_path)
        with open(output_path, 'w') as f:
            json.dump(hostfile, f, indent=2)
        
        logger.info(f"Generated hostfile: {output_path}")
        return output_path
    
    def generate_mpi_hostfile(self, output_path: str = "mpi_hostfile") -> str:
        """Generate MPI-specific hostfile"""
        output_path = os.path.expanduser(output_path)
        with open(output_path, 'w') as f:
            for node in self.cluster_config.nodes:
                # Format: hostname:slots
                f.write(f"{node.ip}:1\n")
        
        logger.info(f"Generated MPI hostfile: {output_path}")
        return output_path
    
    def setup_ssh_keys(self) -> bool:
        """Setup SSH keys for MPI backend"""
        try:
            ssh_dir = Path.home() / ".ssh"
            ssh_dir.mkdir(exist_ok=True, mode=0o700)
            
            # Generate key if doesn't exist
            key_path = ssh_dir / "mlx_cluster_rsa"
            if not key_path.exists():
                logger.info("Generating SSH key for cluster communication")
                subprocess.run([
                    "ssh-keygen", "-t", "rsa", "-b", "4096", "-N", "", 
                    "-f", str(key_path), "-C", "mlx-cluster"
                ], check=True)
                
                # Set proper permissions
                key_path.chmod(0o600)
                (key_path.with_suffix('.pub')).chmod(0o644)
            
            # Configure SSH for cluster
            ssh_config = ssh_dir / "config"
            config_content = f"""
# MLX Cluster Configuration
Host mac-node-*
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    IdentityFile {key_path}
    ConnectTimeout 5
    ServerAliveInterval 60
    ServerAliveCountMax 3
    User {os.getenv('USER', 'mlx')}

"""
            
            # Check if config already exists
            existing_config = ""
            if ssh_config.exists():
                with open(ssh_config, 'r') as f:
                    existing_config = f.read()
            
            # Add config if not already present
            if "MLX Cluster Configuration" not in existing_config:
                with open(ssh_config, 'a') as f:
                    f.write(config_content)
                ssh_config.chmod(0o600)
                logger.info("SSH configuration updated")
            
            # Display public key for manual distribution
            pub_key_path = key_path.with_suffix('.pub')
            if pub_key_path.exists():
                with open(pub_key_path, 'r') as f:
                    pub_key = f.read().strip()
                logger.info(f"Public key for distribution: {pub_key}")
                
            return True
        except Exception as e:
            logger.error(f"SSH setup failed: {e}")
            return False
    
    def test_ssh_connectivity(self) -> Dict[str, bool]:
        """Test SSH connectivity to all nodes"""
        results = {}
        
        if not self.current_node:
            logger.warning("Cannot test SSH connectivity - current node not detected")
            return results
        
        for node in self.cluster_config.nodes:
            if node.name == self.current_node.name:
                continue
                
            try:
                result = subprocess.run([
                    'ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                    f'mlx@{node.ip}', 'echo "ssh_test_success"'
                ], capture_output=True, text=True, timeout=10)
                
                results[node.name] = result.returncode == 0 and 'ssh_test_success' in result.stdout
                
            except subprocess.TimeoutExpired:
                results[node.name] = False
            except Exception as e:
                logger.debug(f"SSH test to {node.name} failed: {e}")
                results[node.name] = False
        
        return results
    
    def test_distributed_setup(self) -> Dict[str, bool]:
        """Test distributed communication setup"""
        results = {}
        
        # Test MPI if available
        try:
            result = subprocess.run(['mpirun', '--version'], 
                                  capture_output=True, text=True)
            results['mpi_available'] = result.returncode == 0
            if results['mpi_available']:
                logger.info("MPI is available")
        except FileNotFoundError:
            results['mpi_available'] = False
            logger.warning("MPI not found")
        
        # Test MLX distributed import
        try:
            import mlx.distributed as dist
            results['mlx_distributed'] = True
            logger.info("MLX distributed module imported successfully")
        except ImportError as e:
            results['mlx_distributed'] = False
            logger.error(f"MLX distributed import failed: {e}")
        
        # Test MLX core functionality
        try:
            import mlx.core as mx
            # Try to create a simple array to test GPU functionality
            arr = mx.array([1, 2, 3])
            results['mlx_core'] = True
            logger.info("MLX core functionality verified")
        except Exception as e:
            results['mlx_core'] = False
            logger.error(f"MLX core test failed: {e}")
        
        # Test network connectivity
        if self.current_node:
            reachable_nodes = 0
            total_nodes = 0
            for node in self.cluster_config.nodes:
                if node.name != self.current_node.name:
                    total_nodes += 1
                    try:
                        result = subprocess.run(['ping', '-c', '1', '-W', '2000', node.ip],
                                              capture_output=True, timeout=5)
                        if result.returncode == 0:
                            reachable_nodes += 1
                            logger.debug(f"Node {node.name} ({node.ip}) is reachable")
                    except subprocess.TimeoutExpired:
                        logger.debug(f"Ping to {node.name} ({node.ip}) timed out")
                    except Exception as e:
                        logger.debug(f"Ping to {node.name} failed: {e}")
            
            results['network_connectivity'] = reachable_nodes == total_nodes if total_nodes > 0 else False
            logger.info(f"Network connectivity: {reachable_nodes}/{total_nodes} nodes reachable")
        else:
            results['network_connectivity'] = False
        
        # Test SSH connectivity
        ssh_results = self.test_ssh_connectivity()
        results['ssh_connectivity'] = all(ssh_results.values()) if ssh_results else False
        
        return results
    
    def get_node_rank(self, node_name: Optional[str] = None) -> int:
        """Get the rank (index) of a node in the cluster"""
        target_node = node_name or (self.current_node.name if self.current_node else None)
        if not target_node:
            return -1
        
        for i, node in enumerate(self.cluster_config.nodes):
            if node.name == target_node:
                return i
        return -1
    
    def get_cluster_size(self) -> int:
        """Get total number of nodes in cluster"""
        return len(self.cluster_config.nodes)
    
    def get_peer_nodes(self) -> List[NodeConfig]:
        """Get list of peer nodes (excluding current node)"""
        if not self.current_node:
            return self.cluster_config.nodes
        
        return [node for node in self.cluster_config.nodes if node.name != self.current_node.name]
    
    def create_distributed_launch_script(self, script_path: str = "launch_distributed.sh") -> str:
        """Create a script to launch distributed MLX processes"""
        script_path = os.path.expanduser(script_path)
        
        script_content = f"""#!/bin/bash
# MLX Distributed Launch Script
# Generated automatically by MLXDistributedConfig

set -e

# Configuration
HOSTFILE="$(pwd)/cluster_hostfile.json"
MPI_HOSTFILE="$(pwd)/mpi_hostfile"
BACKEND="{self.cluster_config.backend}"
TOTAL_RANKS={len(self.cluster_config.nodes)}

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

log() {{
    echo -e "${{GREEN}}[INFO]${{NC}} $1"
}}

warn() {{
    echo -e "${{YELLOW}}[WARN]${{NC}} $1"
}}

error() {{
    echo -e "${{RED}}[ERROR]${{NC}} $1"
    exit 1
}}

# Check if script argument is provided
if [ -z "$1" ]; then
    error "Usage: $0 <python_script> [args...]"
fi

PYTHON_SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

# Verify script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    error "Python script not found: $PYTHON_SCRIPT"
fi

log "Launching distributed MLX job"
log "Script: $PYTHON_SCRIPT"
log "Backend: $BACKEND"
log "Total ranks: $TOTAL_RANKS"

# Activate MLX environment
if [ -f ~/activate_mlx_env.sh ]; then
    source ~/activate_mlx_env.sh
else
    warn "MLX environment script not found"
fi

# Launch based on backend
case "$BACKEND" in
    "mpi")
        log "Using MPI backend"
        if [ ! -f "$MPI_HOSTFILE" ]; then
            error "MPI hostfile not found: $MPI_HOSTFILE"
        fi
        
        mpirun -np $TOTAL_RANKS -hostfile "$MPI_HOSTFILE" \\
               -x MLX_USE_METAL=1 \\
               -x MLX_GPU_MEMORY_FRACTION=0.8 \\
               python "$PYTHON_SCRIPT" $SCRIPT_ARGS
        ;;
    
    "ring")
        log "Using Ring backend"
        if [ ! -f "$HOSTFILE" ]; then
            error "Hostfile not found: $HOSTFILE"
        fi
        
        # For ring backend, we need to launch on each node
        # This is a simplified version - in practice, you'd use a proper orchestrator
        python -m mlx.distributed.launch \\
               --hostfile "$HOSTFILE" \\
               --backend ring \\
               "$PYTHON_SCRIPT" $SCRIPT_ARGS
        ;;
    
    *)
        error "Unknown backend: $BACKEND"
        ;;
esac

log "Distributed job completed"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created distributed launch script: {script_path}")
        return script_path

# Usage example and testing
if __name__ == "__main__":
    import sys
    
    config = MLXDistributedConfig()
    
    print(f"Current node: {config.current_node.name if config.current_node else 'Unknown'}")
    print(f"Cluster backend: {config.cluster_config.backend}")
    print(f"Total nodes: {config.get_cluster_size()}")
    
    if config.current_node:
        print(f"Node rank: {config.get_node_rank()}")
        peer_nodes = config.get_peer_nodes()
        print(f"Peer nodes: {[node.name for node in peer_nodes]}")
    
    # Generate hostfiles
    hostfile_path = config.generate_hostfile()
    print(f"Generated JSON hostfile: {hostfile_path}")
    
    mpi_hostfile_path = config.generate_mpi_hostfile()
    print(f"Generated MPI hostfile: {mpi_hostfile_path}")
    
    # Setup SSH keys
    if config.setup_ssh_keys():
        print("✓ SSH keys configured")
    else:
        print("✗ SSH key setup failed")
    
    # Create launch script
    launch_script = config.create_distributed_launch_script()
    print(f"Created launch script: {launch_script}")
    
    # Test setup
    print("\\nTesting distributed setup...")
    test_results = config.test_distributed_setup()
    print("Test results:")
    for test, result in test_results.items():
        status = '✓' if result else '✗'
        print(f"  {test}: {status}")
    
    # If all tests pass, show next steps
    if all(test_results.values()):
        print(f"\\n{config.current_node.name if config.current_node else 'Node'} is ready for distributed MLX!")
        print("Next steps:")
        print("1. Copy SSH public key to other nodes")
        print("2. Run this script on other nodes to verify setup")
        print("3. Use launch_distributed.sh to run distributed jobs")
    else:
        failed_tests = [test for test, result in test_results.items() if not result]
        print(f"\\nSetup incomplete. Failed tests: {', '.join(failed_tests)}")
        sys.exit(1)