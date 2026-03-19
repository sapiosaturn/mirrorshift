"""Config parsing and merge manager."""

import sys
import tomllib
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Type

import tyro

from mirrorshift.config.job_config import (
    JobConfig,
    validate_activation_checkpoint_config,
    validate_checkpoint_config,
    validate_compile_config,
    validate_data_config,
    validate_debug_config,
    validate_model_config,
    validate_parallelism_config,
    validate_run_config,
    validate_training_config,
)


class ConfigManager:
    """Parse config with precedence: CLI overrides > TOML values > dataclass defaults."""

    def __init__(self, config_cls: Type[JobConfig] = JobConfig):
        self.config_cls = config_cls
        self.config: JobConfig = config_cls()

    def parse_args(self, args: list[str] | None = None) -> JobConfig:
        if args is None:
            args = sys.argv[1:]
        toml_values, config_path = self._maybe_load_toml(args)
        base_config = (
            self._dict_to_dataclass(self.config_cls, toml_values)
            if toml_values is not None
            else self.config_cls()
        )
        parsed = tyro.cli(self.config_cls, args=args, default=base_config)
        self.config = self._normalize_paths(parsed, config_path)
        self._validate_config(self.config)
        return self.config

    def _maybe_load_toml(self, args: list[str]) -> tuple[dict[str, Any] | None, Path]:
        file_path, is_explicit = self._extract_config_path(args)
        path = Path(file_path)
        if not path.exists():
            reason = "Config file does not exist" if is_explicit else "Default config file does not exist"
            raise FileNotFoundError(f"{reason}: {path}")

        with path.open("rb") as file:
            parsed = tomllib.load(file)
        if not isinstance(parsed, dict):
            raise ValueError(f"Invalid TOML root in {path}; expected a table")
        self._resolve_toml_relative_paths(parsed, self._toml_base_dir(path, is_explicit))
        return parsed, path

    def _extract_config_path(self, args: list[str]) -> tuple[str, bool]:
        valid_keys = {"--job.config-file", "--job.config_file"}
        for i, arg in enumerate(args):
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in valid_keys:
                    return value, True
            elif i < len(args) - 1 and arg in valid_keys:
                return args[i + 1], True
        return str(self._default_config_path()), False

    def _default_config_path(self) -> Path:
        return Path(__file__).resolve().parent / "train_configs" / "small.toml"

    def _toml_base_dir(self, config_path: Path, is_explicit: bool) -> Path:
        if is_explicit:
            return config_path.parent
        return Path(__file__).resolve().parents[2]

    def _resolve_toml_relative_paths(self, parsed: dict[str, Any], base_dir: Path) -> None:
        path_fields = {
            ("run", "log_dir"),
            ("data", "snapshot_path"),
            ("data", "plan_path"),
        }
        for section, field_name in path_fields:
            section_data = parsed.get(section)
            if not isinstance(section_data, dict):
                continue
            value = section_data.get(field_name)
            if isinstance(value, str):
                section_data[field_name] = str(self._resolve_path(value, base_dir))

    def _normalize_paths(self, config: JobConfig, config_path: Path) -> JobConfig:
        cwd = Path.cwd()
        return replace(
            config,
            job=replace(
                config.job,
                config_file=str(config_path.resolve()),
            ),
            run=replace(
                config.run,
                log_dir=str(self._resolve_path(config.run.log_dir, cwd)),
            ),
            data=replace(
                config.data,
                snapshot_path=str(self._resolve_path(config.data.snapshot_path, cwd)),
                plan_path=str(self._resolve_path(config.data.plan_path, cwd)),
            ),
        )

    def _resolve_path(self, value: str, base_dir: Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    def _dict_to_dataclass(self, cls: Type[Any], data: dict[str, Any], path: str = "") -> Any:
        if not is_dataclass(cls):
            return data

        field_map = {f.name: f for f in fields(cls)}
        invalid_fields = set(data) - set(field_map)
        if invalid_fields:
            section = path or cls.__name__
            raise ValueError(
                f"Invalid field names in [{section}]: {sorted(invalid_fields)}.\n"
                "Please update your TOML file or override valid fields from the CLI.\n"
                "Run `python -m mirrorshift.train --help` to inspect valid fields."
            )

        values: dict[str, Any] = {}
        for name, dataclass_field in field_map.items():
            if name not in data:
                continue
            value = data[name]
            nested_path = f"{path}.{name}" if path else name
            if is_dataclass(dataclass_field.type):
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected table for [{nested_path}] in TOML, got {type(value).__name__}"
                    )
                values[name] = self._dict_to_dataclass(dataclass_field.type, value, nested_path)
            else:
                values[name] = value

        return cls(**values)

    def _validate_config(self, config: JobConfig) -> None:
        validate_run_config(config.run)
        validate_model_config(config.model)
        validate_training_config(config.training)
        validate_debug_config(config.debug)
        validate_parallelism_config(config.parallelism)
        validate_activation_checkpoint_config(config.activation_checkpoint)
        validate_compile_config(config.compile)
        validate_data_config(config.data)
        validate_checkpoint_config(config.checkpoint)
