"""Run artifact helpers for immutable config snapshots and manifests."""

import json
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mirrorshift.config import JobConfig

RESUME_ALLOWED_CONFIG_DIFFS = {
    "checkpoint.enable",
    "checkpoint.interval",
    "checkpoint.keep_latest_k",
    "checkpoint.load_step",
    "job.config_file",
    "run.wandb_entity",
    "run.wandb_mode",
    "run.wandb_project",
    "training.log_every",
    "training.max_steps",
}
_COMPARE_PATH_FIELDS = {
    "data.plan_path",
    "data.snapshot_path",
    "run.log_dir",
}


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    config_snapshot_path: Path
    manifest_path: Path
    resumed: bool
    staging_dir: Path | None = None

    @property
    def staged(self) -> bool:
        return self.staging_dir is not None


def resolve_run_id(requested_run_id: str | None) -> str:
    if requested_run_id is not None:
        return requested_run_id
    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")


def write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def create_run_artifacts(
    config: JobConfig,
    *,
    resolved_run_id: str | None = None,
    write_files: bool = True,
) -> RunArtifacts:
    run_id = resolved_run_id or resolve_run_id(config.run.id)
    run_dir = Path(config.run.log_dir) / run_id
    staging_dir = _staging_run_dir(run_dir)

    config_snapshot_path = run_dir / config.run.config_snapshot_file
    manifest_path = run_dir / config.run.manifest_file

    resumed = config.checkpoint.load_step is not None
    if resumed:
        if config.run.id is None:
            raise ValueError("run.id must be set when resuming from a checkpoint")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist for resume: {run_dir}")
        if not config_snapshot_path.is_file():
            raise FileNotFoundError(
                f"Missing config snapshot for resume run: {config_snapshot_path}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing run manifest for resume run: {manifest_path}")
        _validate_resume_config(
            requested_config=config,
            config_snapshot_path=config_snapshot_path,
            manifest_path=manifest_path,
        )
    else:
        if write_files:
            if run_dir.exists():
                raise FileExistsError(f"Run directory already exists: {run_dir}")
            if staging_dir.exists():
                raise FileExistsError(f"Staging directory already exists: {staging_dir}")
            staging_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir.mkdir(parents=False, exist_ok=False)
            config_snapshot_path = staging_dir / config.run.config_snapshot_file
            manifest_path = staging_dir / config.run.manifest_file
            write_json_immutable(config_snapshot_path, config.to_dict())
        staging_dir = staging_dir if write_files else None

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        manifest_path=manifest_path,
        resumed=resumed,
        staging_dir=staging_dir,
    )


def finalize_run_artifacts(artifacts: RunArtifacts) -> RunArtifacts:
    if not artifacts.staged:
        return artifacts
    if artifacts.run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {artifacts.run_dir}")

    staging_dir = artifacts.staging_dir
    assert staging_dir is not None
    staging_dir.rename(artifacts.run_dir)
    return RunArtifacts(
        run_id=artifacts.run_id,
        run_dir=artifacts.run_dir,
        config_snapshot_path=artifacts.run_dir / artifacts.config_snapshot_path.name,
        manifest_path=artifacts.run_dir / artifacts.manifest_path.name,
        resumed=artifacts.resumed,
        staging_dir=None,
    )


def discard_staged_run_artifacts(artifacts: RunArtifacts | None) -> None:
    if artifacts is None or not artifacts.staged:
        return
    staging_dir = artifacts.staging_dir
    if staging_dir is not None and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def write_run_manifest(
    artifacts: RunArtifacts,
    config: JobConfig,
    dataset_size: int,
    trainable_params: int,
    device: str,
) -> None:
    payload = {
        "run_id": artifacts.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(artifacts.run_dir),
        "job_config_file": config.job.config_file,
        "spec": config.run.spec,
        "device": device,
        "dataset_size": dataset_size,
        "data_snapshot_path": config.data.snapshot_path,
        "data_plan_path": config.data.plan_path,
        "trainable_params": trainable_params,
        "max_steps": config.training.max_steps,
        "parallelism": {
            "dp_replicate": config.parallelism.dp_replicate,
            "dp_shard": config.parallelism.dp_shard,
            "mixed_precision_param": config.parallelism.mixed_precision_param,
            "mixed_precision_reduce": config.parallelism.mixed_precision_reduce,
            "reshard_after_forward": config.parallelism.reshard_after_forward,
        },
        "wandb_project": config.run.wandb_project,
        "wandb_entity": config.run.wandb_entity,
        "wandb_mode": config.run.wandb_mode,
        "config_snapshot_path": str(artifacts.config_snapshot_path),
        "argv": sys.argv[1:],
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
    }
    write_json_immutable(artifacts.manifest_path, payload)


def _validate_resume_config(
    *,
    requested_config: JobConfig,
    config_snapshot_path: Path,
    manifest_path: Path,
) -> None:
    saved_config = json.loads(config_snapshot_path.read_text())
    if not isinstance(saved_config, dict):
        raise ValueError(f"Invalid saved config snapshot: {config_snapshot_path}")
    manifest_payload = json.loads(manifest_path.read_text())
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Invalid saved run manifest: {manifest_path}")

    saved_flat = _flatten_dict(saved_config)
    current_flat = _flatten_dict(requested_config.to_dict())
    saved_base_dir = Path(str(manifest_payload.get("cwd", Path.cwd())))
    current_base_dir = Path.cwd()

    mismatches: list[str] = []
    for key in sorted(set(saved_flat) | set(current_flat)):
        if key in RESUME_ALLOWED_CONFIG_DIFFS:
            continue
        saved_value = _normalize_resume_value(key, saved_flat.get(key), saved_base_dir)
        current_value = _normalize_resume_value(key, current_flat.get(key), current_base_dir)
        if saved_value != current_value:
            mismatches.append(f"{key}: saved={saved_value!r} requested={current_value!r}")

    if mismatches:
        details = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            details += f"; ... and {len(mismatches) - 8} more"
        raise ValueError(
            "Resume config drift is not allowed outside the explicit allowlist. "
            f"Mismatches: {details}"
        )


def _flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten_dict(value, dotted_key))
        else:
            flattened[dotted_key] = value
    return flattened


def _normalize_resume_value(key: str, value: Any, base_dir: Path) -> Any:
    if key in _COMPARE_PATH_FIELDS and isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())
    return value


def _staging_run_dir(run_dir: Path) -> Path:
    return run_dir.parent / f".{run_dir.name}.staging"
