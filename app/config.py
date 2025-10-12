"""Configuration loader from YAML file."""

import yaml
from pathlib import Path
from typing import Dict
from .models import ModelConfig


class Config:
    """Application configuration."""

    def __init__(self, config_path: str = "config.yml"):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        self.models: Dict[str, ModelConfig] = {}
        self.queue_size: int = 50
        self.request_timeout: int = 600  # 10 minutes
        self.load()

    def load(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Load models
        if 'models' in data:
            for model_name, model_data in data['models'].items():
                self.models[model_name] = ModelConfig(**model_data)

        # Load global settings
        if 'queue_size' in data:
            self.queue_size = data['queue_size']
        if 'request_timeout' in data:
            self.request_timeout = data['request_timeout']

    def get_model(self, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        return self.models[model_name]

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.models.keys())
