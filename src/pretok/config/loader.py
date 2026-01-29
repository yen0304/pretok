"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pretok.config.schema import PretokConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


# Default config file names to search for
CONFIG_FILE_NAMES = [
    "pretok.yaml",
    "pretok.yml",
    "pretok.toml",
    "pretok.json",
    ".pretok.yaml",
    ".pretok.yml",
]


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve environment variable references in config."""
    if isinstance(data, str):
        # Handle ${VAR} syntax
        if data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.environ.get(var_name, data)
        return data
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with path.open() as f:
        data = yaml.safe_load(f)
    result: dict[str, Any] = data if data else {}
    return result


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    with path.open("rb") as f:
        result: dict[str, Any] = tomllib.load(f)
        return result


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON configuration file."""
    import json

    with path.open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load configuration from a file.

    Args:
        path: Path to configuration file (YAML, TOML, or JSON)

    Returns:
        Parsed configuration dictionary

    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix in {".yaml", ".yml"}:
            data = _load_yaml(path)
        elif suffix == ".toml":
            data = _load_toml(path)
        elif suffix == ".json":
            data = _load_json(path)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {suffix}")
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to parse configuration file: {e}") from e

    resolved = _resolve_env_vars(data)
    if not isinstance(resolved, dict):
        return {}
    return resolved


def find_config_file(directory: str | Path | None = None) -> Path | None:
    """Search for a configuration file in the given directory.

    Args:
        directory: Directory to search (defaults to current working directory)

    Returns:
        Path to found configuration file, or None if not found
    """
    directory = Path.cwd() if directory is None else Path(directory)

    for name in CONFIG_FILE_NAMES:
        path = directory / name
        if path.exists():
            return path

    return None


def load_config(
    path: str | Path | None = None,
    *,
    config_dict: dict[str, Any] | None = None,
    auto_discover: bool = True,
) -> PretokConfig:
    """Load and validate pretok configuration.

    Configuration is loaded with the following hierarchy (later overrides earlier):
    1. Built-in defaults
    2. Configuration file (if found/specified)
    3. Runtime config_dict overrides

    Args:
        path: Path to configuration file (optional)
        config_dict: Runtime configuration overrides
        auto_discover: Whether to auto-discover config file if path not specified

    Returns:
        Validated PretokConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    file_config: dict[str, Any] = {}

    # Load from file
    if path is not None:
        file_config = load_config_file(path)
    elif auto_discover:
        found_path = find_config_file()
        if found_path:
            file_config = load_config_file(found_path)

    # Merge with runtime overrides
    if config_dict:
        file_config = _deep_merge(file_config, config_dict)

    # Validate and create config
    try:
        return PretokConfig(**file_config)
    except ValidationError as e:
        error_list = e.errors()
        messages = [f"  - {err['loc']}: {err['msg']}" for err in error_list]
        error_dicts: list[dict[str, Any]] = [dict(err) for err in error_list]
        raise ConfigurationError(
            "Invalid configuration:\n" + "\n".join(messages),
            errors=error_dicts,
        ) from e


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> PretokConfig:
    """Get default configuration with all defaults applied."""
    return PretokConfig()
