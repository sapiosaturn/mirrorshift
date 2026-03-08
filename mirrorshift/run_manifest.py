"""Run artifact helpers for immutable config snapshots and manifests."""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mirrorshift.config import JobConfig


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    config_snapshot_path: Path
    manifest_path: Path
    resumed: bool


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
    else:
        if write_files:
            run_dir.mkdir(parents=True, exist_ok=False)
            write_json_immutable(config_snapshot_path, config.to_dict())
        else:
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
            if not config_snapshot_path.is_file():
                raise FileNotFoundError(f"Missing config snapshot: {config_snapshot_path}")

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        manifest_path=manifest_path,
        resumed=resumed,
    )


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
