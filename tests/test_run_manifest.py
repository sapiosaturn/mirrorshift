import json

import pytest

from mirrorshift.config import CheckpointConfig, JobConfig, Run
from mirrorshift.run_manifest import (
    create_run_artifacts,
    discard_staged_run_artifacts,
    finalize_run_artifacts,
    resolve_run_id,
    write_run_manifest,
)


def test_resolve_run_id_uses_override() -> None:
    assert resolve_run_id("fixed-id") == "fixed-id"


def test_create_run_artifacts_writes_config_snapshot(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="unit-run"))
    artifacts = create_run_artifacts(config)

    assert artifacts.run_dir == tmp_path / "unit-run"
    assert artifacts.staged is True
    assert artifacts.staging_dir == tmp_path / ".unit-run.staging"
    assert artifacts.config_snapshot_path.exists()
    assert not artifacts.manifest_path.exists()
    assert artifacts.resumed is False
    assert not artifacts.run_dir.exists()

    payload = json.loads(artifacts.config_snapshot_path.read_text())
    assert payload["run"]["id"] == "unit-run"
    assert payload["training"]["max_steps"] == 500


def test_create_run_artifacts_fails_for_existing_run_id(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="repeatable"))
    artifacts = create_run_artifacts(config)
    finalize_run_artifacts(artifacts)
    with pytest.raises(FileExistsError):
        create_run_artifacts(config)


def test_create_run_artifacts_reuses_existing_run_dir_for_resume(tmp_path) -> None:
    initial_config = JobConfig(run=Run(log_dir=str(tmp_path), id="resume-run"))
    artifacts = finalize_run_artifacts(create_run_artifacts(initial_config))
    write_run_manifest(
        artifacts=artifacts,
        config=initial_config,
        dataset_size=123,
        trainable_params=456,
        device="cpu",
    )

    resumed_config = JobConfig(
        run=Run(log_dir=str(tmp_path), id="resume-run"),
        checkpoint=CheckpointConfig(enable=True, load_step=-1),
    )
    resumed_artifacts = create_run_artifacts(resumed_config)

    assert resumed_artifacts.run_dir == artifacts.run_dir
    assert resumed_artifacts.config_snapshot_path.read_text() == artifacts.config_snapshot_path.read_text()
    assert resumed_artifacts.resumed is True


def test_create_run_artifacts_allows_resume_safe_config_drift(tmp_path) -> None:
    initial_config = JobConfig(run=Run(log_dir=str(tmp_path), id="resume-safe"))
    artifacts = finalize_run_artifacts(create_run_artifacts(initial_config))
    write_run_manifest(
        artifacts=artifacts,
        config=initial_config,
        dataset_size=123,
        trainable_params=456,
        device="cpu",
    )

    resumed_config = JobConfig(
        run=Run(
            log_dir=str(tmp_path),
            id="resume-safe",
            wandb_mode="offline",
        ),
        training=initial_config.training.__class__(
            **{
                **initial_config.training.__dict__,
                "max_steps": 777,
                "log_every": 3,
            }
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            interval=20,
            keep_latest_k=5,
            load_step=-1,
        ),
    )

    resumed_artifacts = create_run_artifacts(resumed_config)

    assert resumed_artifacts.run_dir == artifacts.run_dir
    assert resumed_artifacts.resumed is True


def test_create_run_artifacts_rejects_resume_config_drift(tmp_path) -> None:
    initial_config = JobConfig(run=Run(log_dir=str(tmp_path), id="resume-drift"))
    artifacts = finalize_run_artifacts(create_run_artifacts(initial_config))
    write_run_manifest(
        artifacts=artifacts,
        config=initial_config,
        dataset_size=123,
        trainable_params=456,
        device="cpu",
    )

    resumed_config = JobConfig(
        run=Run(log_dir=str(tmp_path), id="resume-drift"),
        model=initial_config.model.__class__(
            **{
                **initial_config.model.__dict__,
                "num_layers": initial_config.model.num_layers + 1,
            }
        ),
        checkpoint=CheckpointConfig(enable=True, load_step=-1),
    )

    with pytest.raises(ValueError, match="Resume config drift"):
        create_run_artifacts(resumed_config)


def test_create_run_artifacts_requires_fixed_run_id_for_resume(tmp_path) -> None:
    config = JobConfig(
        run=Run(log_dir=str(tmp_path), id=None),
        checkpoint=CheckpointConfig(enable=True, load_step=-1),
    )
    with pytest.raises(ValueError, match="run.id must be set"):
        create_run_artifacts(config)


def test_write_run_manifest_is_immutable(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="manifest-run"))
    artifacts = finalize_run_artifacts(create_run_artifacts(config))
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
    assert manifest["data_snapshot_path"] == config.data.snapshot_path
    assert manifest["data_plan_path"] == config.data.plan_path
    assert manifest["wandb_project"] == "mirrorshift"
    assert manifest["wandb_mode"] == "online"
    assert manifest["parallelism"]["dp_replicate"] == 1
    assert manifest["parallelism"]["dp_shard"] == 1

    with pytest.raises(FileExistsError):
        write_run_manifest(
            artifacts=artifacts,
            config=config,
            dataset_size=123,
            trainable_params=456,
            device="cpu",
        )


def test_finalize_run_artifacts_promotes_staging_dir(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="promote-run"))
    artifacts = create_run_artifacts(config)

    finalized = finalize_run_artifacts(artifacts)

    assert finalized.staged is False
    assert finalized.run_dir == tmp_path / "promote-run"
    assert finalized.run_dir.is_dir()
    assert finalized.config_snapshot_path == finalized.run_dir / "config.json"
    assert finalized.config_snapshot_path.exists()
    assert not (tmp_path / ".promote-run.staging").exists()


def test_discard_staged_run_artifacts_allows_retry(tmp_path) -> None:
    config = JobConfig(run=Run(log_dir=str(tmp_path), id="retry-run"))
    first = create_run_artifacts(config)

    discard_staged_run_artifacts(first)

    second = create_run_artifacts(config)
    assert second.staged is True
    assert second.config_snapshot_path.exists()
