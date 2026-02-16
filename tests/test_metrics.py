from dataclasses import replace
from pathlib import Path

from mirrorshift.config import JobConfig
from mirrorshift.metrics import NoOpLogger, build_metrics_logger


def test_build_metrics_logger_returns_noop_under_pytest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    logger = build_metrics_logger(
        config=JobConfig(),
        run_id="unit-run",
        run_dir=tmp_path,
    )
    assert isinstance(logger, NoOpLogger)
    logger.log({"train/loss": 1.0}, step=1)
    logger.close()


def test_build_metrics_logger_returns_noop_when_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    config = JobConfig(run=replace(JobConfig().run, wandb_mode="disabled"))
    logger = build_metrics_logger(
        config=config,
        run_id="unit-run",
        run_dir=tmp_path,
    )
    assert isinstance(logger, NoOpLogger)
