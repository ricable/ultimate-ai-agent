import os
import shutil
import subprocess
from typing import Dict, Union, Optional, List
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from sparc_cli.proc.interactive import run_interactive_command
from sparc_cli.text.processing import truncate_output

console = Console()

def install_ripgrep():
    """Install ripgrep using system package manager."""
    try:
        # Try apt-get (Debian/Ubuntu)
        if shutil.which('apt-get'):
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ripgrep'], check=True)
            return True
        # Try yum (RHEL/CentOS)
        elif shutil.which('yum'):
            subprocess.run(['sudo', 'yum', 'install', '-y', 'ripgrep'], check=True)
            return True
        # Try brew (macOS)
        elif shutil.which('brew'):
            subprocess.run(['brew', 'install', 'ripgrep'], check=True)
            return True
        # Try chocolatey (Windows)
        elif shutil.which('choco'):
            subprocess.run(['choco', 'install', '-y', 'ripgrep'], check=True)
            return True
    except subprocess.CalledProcessError:
        return False
    return False

def get_rg_command():
    """Get the ripgrep command, attempting to install if not found."""
    # First check if rg is already in PATH
    system_rg = shutil.which('rg')
    if system_rg:
        return system_rg
    
    # Try to install ripgrep using system package manager
    if install_ripgrep():
        system_rg = shutil.which('rg')
        if system_rg:
            return system_rg
    
    # If system installation fails, use ripgrepy's binary
    try:
        import ripgrepy
        ripgrepy_dir = os.path.dirname(ripgrepy.__file__)
        rg_binary = os.path.join(ripgrepy_dir, 'bin', 'rg')
        
        # Make binary executable if it's not
        if not os.access(rg_binary, os.X_OK):
            os.chmod(rg_binary, 0o755)
        
        return rg_binary
    except ImportError:
        raise RuntimeError("Could not find or install ripgrep. Please install it manually.")


DEFAULT_EXCLUDE_DIRS = [
    '.git',
    'node_modules',
    'vendor',
    '.venv',
    '__pycache__',
    '.cache',
    'dist',
    'build',
    'env',
    '.env',
    'venv',
    '.idea',
    '.vscode'
]

@tool
def ripgrep_search(
    pattern: str,
    *,
    file_type: str = None,
    case_sensitive: bool = True,
    include_hidden: bool = False,
    follow_links: bool = False,
    exclude_dirs: List[str] = None
) -> Dict[str, Union[str, int, bool]]:
    """Execute a ripgrep (rg) search with formatting and common options.

    Args:
        pattern: Search pattern to find
        file_type: Optional file type to filter results (e.g. 'py' for Python files)
        case_sensitive: Whether to do case-sensitive search (default: True)
        include_hidden: Whether to search hidden files and directories (default: False)
        follow_links: Whether to follow symbolic links (default: False)
        exclude_dirs: Additional directories to exclude (combines with defaults)

    Returns:
        Dict containing:
            - output: The formatted search results
            - return_code: Process return code (0 means success)
            - success: Boolean indicating if search succeeded
    """
    # Build rg command with options
    rg_path = get_rg_command()
    cmd = [rg_path, '--color', 'always']
    
    if not case_sensitive:
        cmd.append('-i')
    
    if include_hidden:
        cmd.append('--hidden')
        
    if follow_links:
        cmd.append('--follow')
        
    if file_type:
        cmd.extend(['-t', file_type])

    # Add exclusions
    exclusions = DEFAULT_EXCLUDE_DIRS + (exclude_dirs or [])
    for dir in exclusions:
        cmd.extend(['--glob', f'!{dir}'])

    # Add the search pattern
    cmd.append(pattern)

    # Build info sections for display
    info_sections = []
    
    # Search parameters section
    params = [
        "## Search Parameters",
        f"**Pattern**: `{pattern}`",
        f"**Case Sensitive**: {case_sensitive}",
        f"**File Type**: {file_type or 'all'}"
    ]
    if include_hidden:
        params.append("**Including Hidden Files**: yes")
    if follow_links:
        params.append("**Following Symlinks**: yes")
    if exclude_dirs:
        params.append("\n**Additional Exclusions**:")
        for dir in exclude_dirs:
            params.append(f"- `{dir}`")
    info_sections.append("\n".join(params))

    # Execute command
    console.print(Panel(Markdown(f"Searching for: **{pattern}**"), title="🔎 Ripgrep Search", border_style="bright_blue"))
    try:
        print()
        output, return_code = run_interactive_command(cmd)
        print()
        decoded_output = output.decode() if output else ""
        
        return {
            "output": truncate_output(decoded_output),
            "return_code": return_code,
            "success": return_code == 0
        }
        
    except Exception as e:
        error_msg = str(e)
        console.print(Panel(error_msg, title="❌ Error", border_style="red"))
        return {
            "output": error_msg,
            "return_code": 1,
            "success": False
        }
