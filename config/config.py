"""Configuration management utilities."""
import json
import os
from typing import Dict, Any


class Config:
    """Configuration manager for UAV detection project."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from JSON file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'default.json')
        
        with open(config_path, 'r') as f:
            self._config = json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation support."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key with dot notation support."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            self.set(key, value)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()