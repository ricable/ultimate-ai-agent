# UAP CLI Tool - Command Line Interface for UAP Platform

__version__ = "1.0.0"
__author__ = "UAP Development Team"
__description__ = "Command Line Interface for the Unified Agentic Platform"

from .uap_cli import main, UAPCLIApplication

__all__ = ["main", "UAPCLIApplication"]