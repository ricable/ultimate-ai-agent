#!/usr/bin/env python3
"""
Complete Installation Test

This script tests the entire SAFLA installation structure to ensure
all components work together correctly.
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()


def test_package_structure():
    """Test that all package files exist and are properly structured."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing package structure...[/bold blue]")
    else:
        print("Testing package structure...")
    
    required_files = [
        "pyproject.toml",
        "setup.py", 
        "setup.cfg",
        "requirements.txt",
        "MANIFEST.in",
        "py.typed",
        "safla/__init__.py",
        "safla/cli.py",
        "safla/installer.py",
        "safla/exceptions.py",
        "safla/utils/__init__.py",
        "safla/utils/config.py",
        "safla/utils/logging.py",
        "safla/utils/validation.py",
        "scripts/install.py",
        "scripts/build.py",
        "tests/test_installation.py",
        "docs/INSTALLATION.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Missing files: {missing_files}[/red]")
        else:
            print(f"✗ Missing files: {missing_files}")
        return False
    
    if RICH_AVAILABLE:
        console.print("[green]✓ All required files present[/green]")
    else:
        print("✓ All required files present")
    return True


def test_imports():
    """Test that all modules can be imported correctly."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing module imports...[/bold blue]")
    else:
        print("Testing module imports...")
    
    try:
        # Test main package import
        import safla
        
        # Test submodule imports
        from safla.utils.config import SAFLAConfig
        from safla.utils.logging import setup_logging
        from safla.utils.validation import validate_python_version
        from safla.exceptions import SAFLAError, ConfigurationError
        
        if RICH_AVAILABLE:
            console.print("[green]✓ All modules imported successfully[/green]")
        else:
            print("✓ All modules imported successfully")
        return True
        
    except ImportError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Import error: {e}[/red]")
        else:
            print(f"✗ Import error: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing configuration...[/bold blue]")
    else:
        print("Testing configuration...")
    
    try:
        from safla.utils.config import SAFLAConfig
        
        # Test default configuration
        config = SAFLAConfig()
        assert config.log_level == "INFO"
        assert config.enable_monitoring is True
        
        # Test custom configuration
        custom_config = SAFLAConfig(
            log_level="DEBUG",
            enable_monitoring=False
        )
        assert custom_config.log_level == "DEBUG"
        assert custom_config.enable_monitoring is False
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Configuration management working[/green]")
        else:
            print("✓ Configuration management working")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Configuration error: {e}[/red]")
        else:
            print(f"✗ Configuration error: {e}")
        return False


def test_validation():
    """Test validation functions."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing validation functions...[/bold blue]")
    else:
        print("Testing validation functions...")
    
    try:
        from safla.utils.validation import (
            validate_python_version,
            validate_environment,
            validate_installation
        )
        
        # Test Python version validation
        errors = validate_python_version()
        assert isinstance(errors, list)  # Should return list of errors
        
        # Test environment validation
        env_errors = validate_environment()
        assert isinstance(env_errors, list)
        
        # Test installation validation
        result = validate_installation()
        assert isinstance(result, dict)
        assert "valid" in result
        assert "errors" in result
        assert isinstance(result["valid"], bool)
        assert isinstance(result["errors"], list)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Validation functions working[/green]")
        else:
            print("✓ Validation functions working")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Validation error: {e}[/red]")
        else:
            print(f"✗ Validation error: {e}")
        return False


def test_cli_structure():
    """Test CLI command structure."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing CLI structure...[/bold blue]")
    else:
        print("Testing CLI structure...")
    
    try:
        from safla.cli import cli
        
        # Test that CLI is a Click command
        assert hasattr(cli, 'commands')
        
        # Test that expected commands exist
        expected_commands = ['validate', 'init-config', 'info', 'start', 'status']
        for cmd in expected_commands:
            assert cmd in cli.commands
        
        if RICH_AVAILABLE:
            console.print("[green]✓ CLI structure valid[/green]")
        else:
            print("✓ CLI structure valid")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ CLI structure error: {e}[/red]")
        else:
            print(f"✗ CLI structure error: {e}")
        return False


def test_installer_functionality():
    """Test installer functionality."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing installer functionality...[/bold blue]")
    else:
        print("Testing installer functionality...")
    
    try:
        from safla.installer import SAFLAInstaller
        
        # Create installer instance
        installer = SAFLAInstaller()
        
        # Test that installer has required methods
        assert hasattr(installer, 'run_interactive_install')
        assert hasattr(installer, '_install_dependencies')
        assert hasattr(installer, '_setup_configuration')
        assert hasattr(installer, '_check_prerequisites')
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Installer functionality available[/green]")
        else:
            print("✓ Installer functionality available")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Installer error: {e}[/red]")
        else:
            print(f"✗ Installer error: {e}")
        return False


def test_build_configuration():
    """Test build configuration files."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing build configuration...[/bold blue]")
    else:
        print("Testing build configuration...")
    
    try:
        import tomli
        
        # Test pyproject.toml
        with open("pyproject.toml", "rb") as f:
            pyproject_data = tomli.load(f)
        
        # Verify required sections
        assert "build-system" in pyproject_data
        assert "project" in pyproject_data
        assert "tool" in pyproject_data
        
        # Verify project metadata
        project = pyproject_data["project"]
        assert "name" in project
        assert "version" in project
        assert "description" in project
        assert project["name"] == "safla"
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Build configuration valid[/green]")
        else:
            print("✓ Build configuration valid")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Build configuration error: {e}[/red]")
        else:
            print(f"✗ Build configuration error: {e}")
        return False


def test_package_metadata():
    """Test package metadata and exports."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Testing package metadata...[/bold blue]")
    else:
        print("Testing package metadata...")
    
    try:
        import safla
        
        # Test version
        assert hasattr(safla, '__version__')
        assert isinstance(safla.__version__, str)
        
        # Test author
        assert hasattr(safla, '__author__')
        
        # Test main exports
        expected_exports = ['SAFLAConfig', 'SAFLAError', 'get_logger']
        for export in expected_exports:
            assert hasattr(safla, export)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Package metadata valid[/green]")
        else:
            print("✓ Package metadata valid")
        return True
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Package metadata error: {e}[/red]")
        else:
            print(f"✗ Package metadata error: {e}")
        return False


def run_all_tests():
    """Run all installation tests."""
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold blue]SAFLA Complete Installation Test[/bold blue]\n"
            "Testing all components of the modular installation structure.",
            title="Installation Test",
            border_style="blue"
        ))
    else:
        print("SAFLA Complete Installation Test")
        print("Testing all components of the modular installation structure.")
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Validation", test_validation),
        ("CLI Structure", test_cli_structure),
        ("Installer Functionality", test_installer_functionality),
        ("Build Configuration", test_build_configuration),
        ("Package Metadata", test_package_metadata),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Display results
    if RICH_AVAILABLE:
        table = Table(title="Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        for test_name, passed, error in results:
            status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
            details = error if error else ""
            table.add_row(test_name, status, details)
        
        console.print(table)
    else:
        print("\nTest Results:")
        for test_name, passed, error in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status}")
            if error:
                print(f"    Error: {error}")
    
    # Summary
    passed_tests = sum(1 for _, passed, _ in results if passed)
    total_tests = len(results)
    
    if RICH_AVAILABLE:
        if passed_tests == total_tests:
            console.print(Panel(
                f"[bold green]All tests passed! ({passed_tests}/{total_tests})[/bold green]\n\n"
                "The SAFLA installation structure is working correctly.\n"
                "You can now build and distribute the package.",
                title="Success",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold red]{total_tests - passed_tests} tests failed ({passed_tests}/{total_tests} passed)[/bold red]\n\n"
                "Please fix the failing tests before proceeding.",
                title="Test Failures",
                border_style="red"
            ))
    else:
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
        if passed_tests == total_tests:
            print("All tests passed! The installation structure is working correctly.")
        else:
            print("Some tests failed. Please fix the issues before proceeding.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)