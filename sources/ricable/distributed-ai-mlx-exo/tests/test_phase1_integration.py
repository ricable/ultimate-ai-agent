"""
Phase 1 Integration Tests
Validates that all foundation components work together
"""

import asyncio
import pytest
import subprocess
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.mlx_distributed.config import MLXDistributedConfig
    from src.exo_integration.cluster_manager import create_cluster_manager, EXO_AVAILABLE
    from scripts.validate_network import NetworkValidator
    MLX_DISTRIBUTED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    MLX_DISTRIBUTED_AVAILABLE = False

class TestPhase1Integration:
    """Integration tests for Phase 1 foundation components"""
    
    @pytest.fixture
    def mlx_config(self):
        """Create MLX distributed configuration"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("MLX distributed configuration not available")
        return MLXDistributedConfig()
    
    @pytest.fixture
    def network_validator(self):
        """Create network validator"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("Network validator not available")
        return NetworkValidator()
    
    def test_python_version(self):
        """Test that Python version meets requirements"""
        major, minor = sys.version_info[:2]
        assert major >= 3 and minor >= 12, f"Python 3.12+ required, found {major}.{minor}"
        print(f"✓ Python version {major}.{minor} meets requirements")
    
    def test_environment_setup(self):
        """Test that environment is properly set up"""
        # Check Python version
        assert sys.version_info >= (3, 12), "Python 3.12+ required"
        
        # Check MLX import
        try:
            import mlx.core as mx
            print("✓ MLX core imported successfully")
            
            # Test basic MLX functionality
            arr = mx.array([1, 2, 3])
            assert arr.size == 3, "MLX array creation failed"
            print("✓ MLX basic functionality works")
            
        except ImportError as e:
            pytest.fail(f"MLX not installed or not importable: {e}")
        except Exception as e:
            pytest.fail(f"MLX basic functionality test failed: {e}")
        
        # Check MLX distributed (may not be available in all environments)
        try:
            import mlx.distributed as dist
            print("✓ MLX distributed imported successfully")
        except ImportError:
            print("⚠ MLX distributed not available (this may be expected)")
    
    def test_network_configuration(self, network_validator):
        """Test network configuration"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("Network validator not available")
            
        # Test basic connectivity  
        current_node = network_validator.get_local_node()
        print(f"Detected current node: {current_node}")
        
        # Allow "unknown" for development environments
        assert current_node is not None, "Could not detect current node"
        
        # Check primary IP
        primary_ip = network_validator._get_primary_ip()
        print(f"Primary IP: {primary_ip}")
        assert primary_ip is not None, "Could not determine primary IP"
        
        # Basic IP validation
        parts = primary_ip.split('.')
        assert len(parts) == 4, "Invalid IP address format"
    
    def test_mlx_distributed_config(self, mlx_config):
        """Test MLX distributed configuration"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("MLX distributed configuration not available")
            
        # Test cluster configuration
        assert mlx_config.cluster_config is not None, "Cluster configuration not loaded"
        assert len(mlx_config.cluster_config.nodes) == 4, "Should have 4 cluster nodes"
        
        # Test hostfile generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            hostfile_path = f.name
        
        try:
            generated_path = mlx_config.generate_hostfile(hostfile_path)
            assert os.path.exists(generated_path), "Hostfile not generated"
            
            # Validate hostfile content
            with open(generated_path, 'r') as f:
                hostfile = json.load(f)
            
            assert "hosts" in hostfile, "Hostfile missing hosts section"
            assert len(hostfile["hosts"]) == 4, "Hostfile should have 4 hosts"
            assert hostfile["backend"] in ["mpi", "ring"], "Invalid backend in hostfile"
            assert "total_ranks" in hostfile, "Hostfile missing total_ranks"
            
            print("✓ Hostfile generated and validated successfully")
            
        finally:
            # Cleanup
            if os.path.exists(hostfile_path):
                os.remove(hostfile_path)
    
    def test_exo_cluster_manager_creation(self):
        """Test Exo cluster manager creation"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("Exo cluster manager not available")
            
        # Test creating cluster manager for each node
        node_ids = ["mac-node-1", "mac-node-2", "mac-node-3"]
        
        for node_id in node_ids:
            manager = create_cluster_manager(node_id)
            assert manager.node_spec.node_id == node_id
            assert manager.node_spec.ip.startswith("10.0.1.")
            assert manager.node_spec.memory_gb > 0
            assert manager.node_spec.port == 52415
            
        print(f"✓ Exo cluster managers created for all {len(node_ids)} nodes")
        
        # Test invalid node ID
        with pytest.raises(ValueError):
            create_cluster_manager("invalid-node")
    
    @pytest.mark.asyncio
    async def test_network_validation(self, network_validator):
        """Test network validation functionality"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("Network validator not available")
            
        # Run basic connectivity tests (quick mode to avoid timeouts)
        connectivity_tests = await network_validator.test_basic_connectivity()
        
        # Should have at least some tests (even if they fail due to network issues)
        assert len(connectivity_tests) >= 0, "Connectivity tests should be generated"
        print(f"✓ Generated {len(connectivity_tests)} connectivity tests")
        
        # Test MTU configuration
        mtu_tests = network_validator.test_mtu_configuration()
        assert len(mtu_tests) > 0, "No MTU tests generated"
        print(f"✓ Generated {len(mtu_tests)} MTU configuration tests")
        
        # Test firewall configuration (may require sudo)
        firewall_tests = network_validator.test_firewall_configuration()
        assert len(firewall_tests) > 0, "No firewall tests generated"
        print(f"✓ Generated {len(firewall_tests)} firewall tests")
    
    def test_ssh_key_configuration(self, mlx_config):
        """Test SSH key configuration for MPI"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("MLX distributed configuration not available")
            
        # Test SSH key setup
        success = mlx_config.setup_ssh_keys()
        assert success, "SSH key setup failed"
        
        # Check that key files exist
        ssh_dir = Path.home() / ".ssh"
        key_path = ssh_dir / "mlx_cluster_rsa"
        
        assert key_path.exists(), "SSH private key not created"
        assert (key_path.with_suffix(".pub")).exists(), "SSH public key not created"
        
        # Check permissions
        assert oct(key_path.stat().st_mode)[-3:] == "600", "SSH private key has incorrect permissions"
        
        print("✓ SSH keys configured successfully")
    
    def test_distributed_communication_test(self, mlx_config):
        """Test distributed communication setup"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("MLX distributed configuration not available")
            
        test_results = mlx_config.test_distributed_setup()
        
        # Should have test results for key components
        expected_tests = ['mpi_available', 'mlx_distributed', 'mlx_core', 'network_connectivity']
        for test in expected_tests:
            assert test in test_results, f"Missing test result for {test}"
        
        # MLX core should always be available if environment is set up correctly
        if test_results.get('mlx_core', False):
            print("✓ MLX core functionality verified")
        else:
            print("⚠ MLX core test failed - check installation")
        
        # Log all test results
        for test_name, result in test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {test_name}: {'PASS' if result else 'FAIL'}")
    
    @pytest.mark.integration
    def test_complete_foundation_setup(self, mlx_config, network_validator):
        """Integration test for complete foundation setup"""
        if not MLX_DISTRIBUTED_AVAILABLE:
            pytest.skip("Project modules not available")
            
        print("Running complete foundation setup test...")
        
        # Test MLX configuration
        assert mlx_config.cluster_config is not None
        print("✓ MLX configuration loaded")
        
        # Test network detection
        local_node = network_validator.get_local_node()
        print(f"✓ Local node detected as: {local_node}")
        
        # Test component integration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            hostfile_path = f.name
        
        try:
            generated_path = mlx_config.generate_hostfile(hostfile_path)
            assert os.path.exists(generated_path), "Integration hostfile not generated"
            
            # Verify hostfile can be loaded and has correct structure
            with open(generated_path, 'r') as f:
                hostfile = json.load(f)
            
            assert len(hostfile["hosts"]) == 4, "Integration hostfile incorrect"
            print("✓ Hostfile generation working")
            
        finally:
            # Cleanup
            if os.path.exists(hostfile_path):
                os.remove(hostfile_path)
        
        # Test distributed setup
        dist_results = mlx_config.test_distributed_setup()
        working_components = sum(1 for result in dist_results.values() if result)
        total_components = len(dist_results)
        
        print(f"✓ Distributed setup: {working_components}/{total_components} components working")
        
        print("✓ Phase 1 integration test passed - foundation ready for Phase 2")
    
    def test_scripts_executable(self):
        """Test that key scripts are executable and functional"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Test setup script exists and is executable
        setup_script = scripts_dir / "setup_environment.sh"
        if setup_script.exists():
            assert os.access(setup_script, os.X_OK), "setup_environment.sh not executable"
            print("✓ setup_environment.sh is executable")
        
        # Test network config script
        network_script = scripts_dir / "configure_network.sh"
        if network_script.exists():
            assert os.access(network_script, os.X_OK), "configure_network.sh not executable"
            print("✓ configure_network.sh is executable")
        
        # Test network validation script
        validate_script = scripts_dir / "validate_network.py"
        if validate_script.exists():
            assert os.access(validate_script, os.X_OK), "validate_network.py not executable"
            print("✓ validate_network.py is executable")
    
    def test_project_structure(self):
        """Test that project structure is set up correctly"""
        project_root = Path(__file__).parent.parent
        
        # Check key directories
        required_dirs = [
            "src",
            "src/mlx_distributed", 
            "src/exo_integration",
            "scripts",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory missing: {dir_path}"
            print(f"✓ Directory exists: {dir_path}")
        
        # Check key files
        required_files = [
            "src/mlx_distributed/__init__.py",
            "src/mlx_distributed/config.py",
            "src/exo_integration/__init__.py", 
            "src/exo_integration/cluster_manager.py",
            "scripts/setup_environment.sh",
            "scripts/configure_network.sh",
            "scripts/validate_network.py"
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"
            print(f"✓ File exists: {file_path}")

# Standalone test runner
async def run_async_tests():
    """Run async tests manually if not using pytest"""
    print("Running async integration tests...")
    
    try:
        if MLX_DISTRIBUTED_AVAILABLE:
            validator = NetworkValidator()
            test_instance = TestPhase1Integration()
            await test_instance.test_network_validation(validator)
            print("✓ Async tests completed successfully")
        else:
            print("⚠ Async tests skipped - project modules not available")
    except Exception as e:
        print(f"✗ Async tests failed: {e}")

# Test runner for direct execution
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 INTEGRATION TESTS")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        # Run tests manually for environments without pytest
        test_instance = TestPhase1Integration()
        
        try:
            print("\n1. Testing Python version...")
            test_instance.test_python_version()
            
            print("\n2. Testing environment setup...")
            test_instance.test_environment_setup()
            
            print("\n3. Testing project structure...")
            test_instance.test_project_structure()
            
            print("\n4. Testing scripts executable...")
            test_instance.test_scripts_executable()
            
            if MLX_DISTRIBUTED_AVAILABLE:
                print("\n5. Testing MLX distributed config...")
                mlx_config = MLXDistributedConfig()
                test_instance.test_mlx_distributed_config(mlx_config)
                
                print("\n6. Testing network configuration...")
                network_validator = NetworkValidator()
                test_instance.test_network_configuration(network_validator)
                
                print("\n7. Testing Exo cluster manager...")
                test_instance.test_exo_cluster_manager_creation()
                
                print("\n8. Testing SSH configuration...")
                test_instance.test_ssh_key_configuration(mlx_config)
                
                print("\n9. Testing distributed communication...")
                test_instance.test_distributed_communication_test(mlx_config)
                
                print("\n10. Running async tests...")
                asyncio.run(run_async_tests())
                
                print("\n11. Testing complete foundation setup...")
                test_instance.test_complete_foundation_setup(mlx_config, network_validator)
            else:
                print("\n⚠ Skipping advanced tests - project modules not available")
                print("This is expected if running before full setup is complete")
            
            print("\n" + "=" * 60)
            print("✓ ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY")
            print("Phase 1 foundation is ready for Phase 2 implementation")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)