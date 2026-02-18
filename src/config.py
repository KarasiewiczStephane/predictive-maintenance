"""Centralized configuration management using Pydantic Settings.

Provides type-safe configuration with environment variable support,
validation, and YAML config file loading for different environments.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable overrides.

    Settings are loaded in order of priority:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("data"))
    model_dir: Path = Field(default=Path("models"))

    # Model settings
    model_type: str = Field(default="random_forest")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=10)

    # Feature engineering
    window_sizes: str = Field(default="5,10,30")
    missing_value_strategy: str = Field(default="interpolate")

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")
    log_json_format: bool = Field(default=False)

    def get_window_sizes(self) -> list[int]:
        """Parse window sizes from comma-separated string.

        Returns:
            List of integer window sizes for rolling statistics.
        """
        return [int(w.strip()) for w in self.window_sizes.split(",")]


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_settings(env: Optional[str] = None) -> Settings:
    """Create Settings instance, optionally loading from a YAML env file.

    Args:
        env: Environment name (e.g., 'dev', 'prod'). If provided,
            loads configs/{env}.yaml and uses values as defaults.

    Returns:
        Configured Settings instance.
    """
    if env:
        config_path = Path("configs") / f"{env}.yaml"
        try:
            yaml_config = load_yaml_config(config_path)
            return Settings(**yaml_config)
        except FileNotFoundError:
            logger.warning("Config file %s not found, using defaults", config_path)
    return Settings()


settings = get_settings()
