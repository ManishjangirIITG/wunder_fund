from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class ExperimentConfig:
    experiment_name: str
    model_name: str
    model_version: str
    dataset_path: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """File-backed experiment registry."""

    def __init__(self, experiments_dir: str | Path) -> None:
        self.root = Path(experiments_dir)
        self.runs_dir = self.root / "runs"
        self.registry = self.root / "registry.jsonl"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, config: ExperimentConfig) -> str:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        run_dir = self.runs_dir / run_id
        run_dir.mkdir()

        dataset = Path(config.dataset_path)
        record = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "status": "RUNNING",
            "config": asdict(config),
            "dataset": {
                "path": str(dataset),
                "sha256": sha256_file(dataset),
                "size_bytes": (
                    dataset.stat().st_size if dataset.exists() else None
                ),
            },
        }
        self._dump(run_dir / "config.json", record)

        self._dump(
            run_dir / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": git_commit(),
            },
        )
        return run_id

    def finish_run(
        self,
        run_id: str,
        metrics: dict[str, Any],
        status: str = "COMPLETED",
        notes: str = "",
    ) -> None:
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(run_id)

        finished = utc_now()
        self._dump(
            run_dir / "metrics.json",
            {
                "run_id": run_id,
                "finished_at_utc": finished,
                "status": status,
                "metrics": metrics,
                "notes": notes,
            },
        )

        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["status"] = status
        config["finished_at_utc"] = finished
        self._dump(config_path, config)

        with self.registry.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "experiment_name": config["config"]["experiment_name"],
                        "model_name": config["config"]["model_name"],
                        "model_version": config["config"]["model_version"],
                        "status": status,
                        "mean_r2": metrics.get("mean_r2"),
                        "finished_at_utc": finished,
                    }
                )
                + "\n"
            )

    @staticmethod
    def _dump(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
