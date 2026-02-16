import json

import pytest

from mirrorshift.config import JobConfig, Run
from mirrorshift.run_manifest import (
    create_run_artifacts,
    resolve_run_id,
    write_run_manifest,
)


def test_resolve_run_id_uses_override() -> None:
    assert resolve_run_id("fixed-id") == "fixed-id"


def test_create_run_artifacts_writes_config_snapshot(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="unit-run"))
    artifacts = create_run_artifacts(config)

    assert artifacts.run_dir == tmp_path / "unit-run"
    assert artifacts.config_snapshot_path.exists()
    assert not artifacts.manifest_path.exists()

    payload = json.loads(artifacts.config_snapshot_path.read_text())
    assert payload["run"]["id"] == "unit-run"
    assert payload["training"]["max_steps"] == 500


def test_create_run_artifacts_fails_for_existing_run_id(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="repeatable"))
    create_run_artifacts(config)
    with pytest.raises(FileExistsError):
        create_run_artifacts(config)


def test_write_run_manifest_is_immutable(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="manifest-run"))
    artifacts = create_run_artifacts(config)
    write_run_manifest(
        artifacts=artifacts,
        config=config,
        dataset_size=123,
        trainable_params=456,
        device="cpu",
    )

    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["run_id"] == "manifest-run"
    assert manifest["dataset_size"] == 123
    assert manifest["trainable_params"] == 456
    assert manifest["device"] == "cpu"
    assert manifest["wandb_project"] == "mirrorshift"
    assert manifest["wandb_mode"] == "online"

    with pytest.raises(FileExistsError):
        write_run_manifest(
            artifacts=artifacts,
            config=config,
            dataset_size=123,
            trainable_params=456,
            device="cpu",
        )
