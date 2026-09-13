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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
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
    """Small file-based experiment registry.

    Each run creates:
        experiments/runs/<run_id>/config.json
        experiments/runs/<run_id>/metrics.json
        experiments/runs/<run_id>/environment.json

    The files are immutable records of what was executed and what it produced.
    """

    def __init__(self, experiments_dir: str | Path) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.runs_dir = self.experiments_dir / "runs"
        self.registry_path = self.experiments_dir / "registry.jsonl"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, config: ExperimentConfig) -> str:
        run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=False)

        dataset_path = Path(config.dataset_path)
        dataset_info = {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
            "size_bytes": dataset_path.stat().st_size if dataset_path.exists() else None,
        }

        run_config = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "status": "RUNNING",
            "config": asdict(config),
            "dataset": dataset_info,
        }
        self._write_json(run_dir / "config.json", run_config)

        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": get_git_commit(),
        }
        self._write_json(run_dir / "environment.json", environment)

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
            raise FileNotFoundError(f"Unknown run_id: {run_id}")

        metric_record = {
            "run_id": run_id,
            "finished_at_utc": utc_now(),
            "status": status,
            "metrics": metrics,
            "notes": notes,
        }
        self._write_json(run_dir / "metrics.json", metric_record)

        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["status"] = status
        config["finished_at_utc"] = metric_record["finished_at_utc"]
        self._write_json(config_path, config)

        registry_record = {
            "run_id": run_id,
            "experiment_name": config["config"]["experiment_name"],
            "model_name": config["config"]["model_name"],
            "model_version": config["config"]["model_version"],
            "status": status,
            "mean_r2": metrics.get("mean_r2"),
            "finished_at_utc": metric_record["finished_at_utc"],
        }
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(registry_record) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
