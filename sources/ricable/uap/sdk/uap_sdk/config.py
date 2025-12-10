# UAP SDK Configuration Module
"""
Configuration management for UAP SDK.
Handles loading and storing configuration from files, environment variables, and code.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

DEFAULT_CONFIG = {
    "backend_url": "http://localhost:8000",
    "websocket_url": "ws://localhost:8000", 
    "http_timeout": 30,
    "websocket_timeout": 30,
    "use_websocket": False,
    "log_level": "INFO",
    "max_retries": 3,
    "retry_delay": 1.0,
    "max_conversation_history": 50,
    "auto_reconnect": True
}


class Configuration:
    """Configuration manager for UAP SDK."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None, config_dict: Optional[Dict[str, Any]] = None):
        self._config = DEFAULT_CONFIG.copy()
        self._config_file = None
        
        # Load from config file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load from environment variables
        self._load_from_env()
        
        # Override with provided config dict
        if config_dict:
            self._config.update(config_dict)
    
    def load_from_file(self, config_file: Union[str, Path]) -> None:
        """Load configuration from a file (JSON or YAML)."""
        config_path = Path(config_file)
        self._config_file = config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            if file_config:
                self._config.update(file_config)
                
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "UAP_BACKEND_URL": "backend_url",
            "UAP_WEBSOCKET_URL": "websocket_url", 
            "UAP_HTTP_TIMEOUT": ("http_timeout", int),
            "UAP_WEBSOCKET_TIMEOUT": ("websocket_timeout", int),
            "UAP_USE_WEBSOCKET": ("use_websocket", lambda x: x.lower() in ('true', '1', 'yes')),
            "UAP_LOG_LEVEL": "log_level",
            "UAP_MAX_RETRIES": ("max_retries", int),
            "UAP_RETRY_DELAY": ("retry_delay", float),
            "UAP_MAX_CONVERSATION_HISTORY": ("max_conversation_history", int),
            "UAP_AUTO_RECONNECT": ("auto_reconnect", lambda x: x.lower() in ('true', '1', 'yes')),
            "UAP_ACCESS_TOKEN": "access_token",
            "UAP_REFRESH_TOKEN": "refresh_token"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    try:
                        self._config[key] = converter(value)
                    except (ValueError, TypeError):
                        # Skip invalid conversions
                        pass
                else:
                    self._config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def remove(self, key: str) -> None:
        """Remove a configuration value."""
        self._config.pop(key, None)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with a dictionary."""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as a dictionary."""
        return self._config.copy()
    
    def save_to_file(self, config_file: Optional[Union[str, Path]] = None, format: str = "json") -> None:
        """Save configuration to a file."""
        if config_file:
            output_path = Path(config_file)
        elif self._config_file:
            output_path = self._config_file
        else:
            raise ValueError("No config file specified")
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                if format.lower() == "yaml" or output_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2)
                    
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {output_path}: {str(e)}")
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict-like syntax."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dict-like syntax."""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self._config
    
    def validate(self) -> bool:
        """Validate the configuration."""
        required_keys = ["backend_url", "websocket_url"]
        
        for key in required_keys:
            if not self._config.get(key):
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate URLs
        backend_url = self._config.get("backend_url", "")
        if not (backend_url.startswith("http://") or backend_url.startswith("https://")):
            raise ValueError("backend_url must start with http:// or https://")
        
        websocket_url = self._config.get("websocket_url", "")
        if not (websocket_url.startswith("ws://") or websocket_url.startswith("wss://")):
            raise ValueError("websocket_url must start with ws:// or wss://")
        
        # Validate numeric values
        numeric_keys = ["http_timeout", "websocket_timeout", "max_retries", "retry_delay", "max_conversation_history"]
        for key in numeric_keys:
            value = self._config.get(key)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"Configuration key {key} must be numeric")
            if value is not None and value <= 0:
                raise ValueError(f"Configuration key {key} must be positive")
        
        return True
    
    @classmethod
    def from_env(cls) -> 'Configuration':
        """Create configuration from environment variables only."""
        config = cls()
        config._config = DEFAULT_CONFIG.copy()
        config._load_from_env()
        return config
    
    @classmethod
    def create_default_config_file(cls, config_file: Union[str, Path], format: str = "json") -> None:
        """Create a default configuration file."""
        config = cls()
        config.save_to_file(config_file, format)


class ConfigurationProfile:
    """Manage multiple configuration profiles."""
    
    def __init__(self, profiles_dir: Union[str, Path] = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else Path.home() / ".uap" / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.current_profile = "default"
    
    def create_profile(self, name: str, config: Configuration) -> None:
        """Create a new configuration profile."""
        profile_file = self.profiles_dir / f"{name}.json"
        config.save_to_file(profile_file)
    
    def load_profile(self, name: str) -> Configuration:
        """Load a configuration profile."""
        profile_file = self.profiles_dir / f"{name}.json"
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile '{name}' not found")
        
        return Configuration(config_file=profile_file)
    
    def list_profiles(self) -> list:
        """List available profiles."""
        profiles = []
        for file in self.profiles_dir.glob("*.json"):
            profiles.append(file.stem)
        return sorted(profiles)
    
    def delete_profile(self, name: str) -> None:
        """Delete a configuration profile."""
        if name == "default":
            raise ValueError("Cannot delete default profile")
        
        profile_file = self.profiles_dir / f"{name}.json"
        if profile_file.exists():
            profile_file.unlink()
    
    def set_current_profile(self, name: str) -> None:
        """Set the current active profile."""
        profile_file = self.profiles_dir / f"{name}.json"
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile '{name}' not found")
        
        self.current_profile = name
        
        # Save current profile setting
        current_file = self.profiles_dir / ".current"
        with open(current_file, 'w') as f:
            f.write(name)
    
    def get_current_profile(self) -> str:
        """Get the current active profile."""
        current_file = self.profiles_dir / ".current"
        if current_file.exists():
            with open(current_file, 'r') as f:
                return f.read().strip()
        return "default"
    
    def get_current_config(self) -> Configuration:
        """Get configuration for the current profile."""
        current_profile = self.get_current_profile()
        return self.load_profile(current_profile)