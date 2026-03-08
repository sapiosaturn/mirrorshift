from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mirrordata.cli import main


def _write_parquet(path: Path, texts: list[str]) -> None:
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def test_mirrordata_cli_prep_and_plan(monkeypatch, capsys, tmp_path: Path) -> None:
    parquet_path = tmp_path / "train.parquet"
    snapshot_dir = tmp_path / "snapshot"
    plan_dir = tmp_path / "plan"
    _write_parquet(
        parquet_path,
        [
            "alpha beta gamma delta epsilon zeta eta theta",
            "iota kappa lambda mu nu xi omicron pi",
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "mirrordata",
            "prep-parquet",
            str(parquet_path),
            "--output-dir",
            str(snapshot_dir),
            "--snapshot-id",
            "cli-snapshot",
            "--dataset-name",
            "cli-dataset",
        ],
    )
    assert main() == 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "mirrordata",
            "build-plan",
            "--snapshot-path",
            str(snapshot_dir),
            "--output-dir",
            str(plan_dir),
            "--sequence-length",
            "4",
            "--stride",
            "2",
        ],
    )
    assert main() == 0

    monkeypatch.setattr(
        "sys.argv",
        ["mirrordata", "info", str(snapshot_dir / "manifest.json")],
    )
    assert main() == 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "mirrordata",
            "verify",
            "--snapshot-path",
            str(snapshot_dir),
            "--plan-path",
            str(plan_dir),
        ],
    )
    assert main() == 0

    output = capsys.readouterr().out
    assert "cli-snapshot" in output
    assert "wrote snapshot" in output
    assert "wrote plan" in output
    assert "snapshot and plan verified" in output


def test_mirrordata_cli_verify_reports_corruption(monkeypatch, capsys, tmp_path: Path) -> None:
    parquet_path = tmp_path / "train.parquet"
    snapshot_dir = tmp_path / "snapshot"
    _write_parquet(
        parquet_path,
        [
            "alpha beta gamma delta epsilon zeta eta theta",
            "iota kappa lambda mu nu xi omicron pi",
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "mirrordata",
            "prep-parquet",
            str(parquet_path),
            "--output-dir",
            str(snapshot_dir),
            "--snapshot-id",
            "cli-snapshot",
            "--dataset-name",
            "cli-dataset",
        ],
    )
    assert main() == 0

    shard_path = next((snapshot_dir / "shards").glob("*.bin"))
    shard_path.write_bytes(shard_path.read_bytes()[:-4])

    monkeypatch.setattr(
        "sys.argv",
        [
            "mirrordata",
            "verify",
            "--snapshot-path",
            str(snapshot_dir),
        ],
    )
    assert main() == 1

    output = capsys.readouterr().out
    assert "ERROR:" in output
