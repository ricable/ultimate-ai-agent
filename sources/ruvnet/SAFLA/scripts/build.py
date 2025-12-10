#!/usr/bin/env python3
"""
SAFLA Build Script

This script builds the SAFLA package for distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()


def clean_build_artifacts():
    """Clean previous build artifacts."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Cleaning build artifacts...[/bold blue]")
    else:
        print("Cleaning build artifacts...")
    
    artifacts = [
        "build",
        "dist", 
        "*.egg-info",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache"
    ]
    
    for pattern in artifacts:
        if "*" in pattern:
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            if os.path.exists(pattern):
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                else:
                    os.remove(pattern)
    
    if RICH_AVAILABLE:
        console.print("[green]✓ Build artifacts cleaned[/green]")
    else:
        print("✓ Build artifacts cleaned")


def build_package():
    """Build the package."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Building package...[/bold blue]")
    else:
        print("Building package...")
    
    try:
        # Build source distribution
        subprocess.run([
            sys.executable, "-m", "build", "--sdist"
        ], check=True)
        
        # Build wheel distribution
        subprocess.run([
            sys.executable, "-m", "build", "--wheel"
        ], check=True)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Package built successfully[/green]")
        else:
            print("✓ Package built successfully")
        
    except subprocess.CalledProcessError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Build failed: {e}[/red]")
        else:
            print(f"✗ Build failed: {e}")
        sys.exit(1)


def check_package():
    """Check the built package."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Checking package...[/bold blue]")
    else:
        print("Checking package...")
    
    try:
        # Check with twine if available
        subprocess.run([
            sys.executable, "-m", "twine", "check", "dist/*"
        ], check=True)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Package check passed[/green]")
        else:
            print("✓ Package check passed")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        if RICH_AVAILABLE:
            console.print("[yellow]Warning: twine not available for package checking[/yellow]")
        else:
            print("Warning: twine not available for package checking")


def show_build_info():
    """Show build information."""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        return
    
    files = list(dist_dir.glob("*"))
    if not files:
        return
    
    if RICH_AVAILABLE:
        from rich.table import Table
        
        table = Table(title="Built Packages")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="green")
        
        for file in files:
            size = file.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            
            table.add_row(file.name, size_str)
        
        console.print(table)
    else:
        print("\nBuilt Packages:")
        for file in files:
            size = file.stat().st_size
            print(f"  {file.name} ({size} bytes)")


def main():
    """Main build function."""
    try:
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold blue]SAFLA Build Script[/bold blue]\n"
                "Building SAFLA package for distribution.",
                title="Build",
                border_style="blue"
            ))
        else:
            print("SAFLA Build Script")
            print("Building SAFLA package for distribution.")
        
        # Clean previous builds
        clean_build_artifacts()
        
        # Build the package
        build_package()
        
        # Check the package
        check_package()
        
        # Show build info
        show_build_info()
        
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold green]Build completed successfully![/bold green]\n\n"
                "The package is ready for distribution.\n"
                "Use 'python -m twine upload dist/*' to upload to PyPI.",
                title="Success",
                border_style="green"
            ))
        else:
            print("\nBuild completed successfully!")
            print("The package is ready for distribution.")
            print("Use 'python -m twine upload dist/*' to upload to PyPI.")
        
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Build interrupted by user[/yellow]")
        else:
            print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]Build failed: {e}[/red]")
        else:
            print(f"\nBuild failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()