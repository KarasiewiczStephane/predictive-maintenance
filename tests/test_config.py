"""Tests for the configuration management module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Settings, get_settings, load_yaml_config


class TestSettings:
    """Tests for Settings class defaults and validation."""

    def test_default_values(self) -> None:
        """Default settings load correctly without any env vars."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)
        assert s.data_dir == Path("data")
        assert s.model_dir == Path("models")
        assert s.model_type == "random_forest"
        assert s.test_size == 0.2
        assert s.random_state == 42
        assert s.api_host == "0.0.0.0"
        assert s.api_port == 8000
        assert s.debug is False
        assert s.log_level == "INFO"

    def test_env_override(self) -> None:
        """Environment variables override default values."""
        env = {
            "MODEL_TYPE": "xgboost",
            "TEST_SIZE": "0.3",
            "API_PORT": "9000",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
        assert s.model_type == "xgboost"
        assert s.test_size == 0.3
        assert s.api_port == 9000
        assert s.debug is True
        assert s.log_level == "DEBUG"

    def test_test_size_validation_too_high(self) -> None:
        """test_size above 0.5 raises validation error."""
        with patch.dict(os.environ, {"TEST_SIZE": "0.9"}, clear=True):
            with pytest.raises(Exception):
                Settings(_env_file=None)

    def test_test_size_validation_too_low(self) -> None:
        """test_size below 0.1 raises validation error."""
        with patch.dict(os.environ, {"TEST_SIZE": "0.05"}, clear=True):
            with pytest.raises(Exception):
                Settings(_env_file=None)

    def test_get_window_sizes(self) -> None:
        """Window sizes parsed correctly from comma-separated string."""
        with patch.dict(os.environ, {"WINDOW_SIZES": "3,7,14"}, clear=True):
            s = Settings(_env_file=None)
        assert s.get_window_sizes() == [3, 7, 14]

    def test_get_window_sizes_default(self) -> None:
        """Default window sizes are 5, 10, 30."""
        with patch.dict(os.environ, {}, clear=True):
            s = Settings(_env_file=None)
        assert s.get_window_sizes() == [5, 10, 30]


class TestLoadYamlConfig:
    """Tests for YAML config loading."""

    def test_load_existing_yaml(self, tmp_path: Path) -> None:
        """YAML config loads correctly."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model_type: xgboost\ntest_size: 0.3\n")
        result = load_yaml_config(config_file)
        assert result["model_type"] == "xgboost"
        assert result["test_size"] == 0.3

    def test_load_missing_yaml(self) -> None:
        """Missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("/nonexistent/config.yaml"))

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        result = load_yaml_config(config_file)
        assert result == {}


class TestGetSettings:
    """Tests for get_settings factory function."""

    def test_get_settings_no_env(self) -> None:
        """get_settings without env returns default settings."""
        with patch.dict(os.environ, {}, clear=True):
            s = get_settings()
        assert s.model_type == "random_forest"

    def test_get_settings_missing_env_file(self) -> None:
        """get_settings with missing env file falls back to defaults."""
        with patch.dict(os.environ, {}, clear=True):
            s = get_settings(env="nonexistent")
        assert s.model_type == "random_forest"
