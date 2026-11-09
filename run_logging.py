"""Shared run directories, structured metrics and log archives."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import time
import uuid


LOG_FILES = ("run.log", "metrics.jsonl", "config.resolved.json", "manifest.json", "summary.json")


def utc_now():
    return datetime.now(timezone.utc)


def sha256_file(path):
    with Path(path).open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


class RunLog:
    """Log one attempt. Use logger for console messages and metric() for data."""

    def __init__(self, stage, runs_dir, archives_dir, config, *, manifest=None):
        if not stage or Path(stage).name != stage or stage in (".", ".."):
            raise ValueError("Stage must be a single directory name.")
        self.started = utc_now()
        suffix = uuid.uuid4().hex[:12]
        self.run_id = self.started.strftime("%Y%m%dT%H%M%S%fZ") + "-" + suffix
        self.path = Path(runs_dir) / stage / self.run_id
        self.archive_path = Path(archives_dir) / stage / f"{self.run_id}.tar.gz"
        self.config = config
        self.manifest = {
            "run_id": self.run_id, "stage": stage, "python": platform.python_version(),
            "platform": platform.platform(), "command": [sys.executable, *sys.argv],
            **(manifest or {}),
        }
        self.summary = {"run_id": self.run_id, "started_at": self.started.isoformat()}
        self.archive_files = list(LOG_FILES)
        self.logger = logging.Logger(stage, level=logging.INFO)

    def __enter__(self):
        self.path.mkdir(parents=True)
        (self.path / "metrics.jsonl").touch()
        write_json(self.path / "config.resolved.json", self.config)
        self.update_manifest()
        write_json(self.path / "summary.json", {**self.summary, "status": "running"})
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        formatter.converter = time.gmtime
        for handler in (logging.StreamHandler(), logging.FileHandler(self.path / "run.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info("Run directory %s", self.path)
        return self

    def update_manifest(self, **fields):
        self.manifest.update(fields)
        write_json(self.path / "manifest.json", self.manifest)

    def metric(self, **fields):
        row = {"timestamp": utc_now().isoformat(), **fields}
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def include_in_archive(self, filename):
        """Include an additional run-local report by its basename."""
        if Path(filename).name != filename or not (self.path / filename).is_file():
            raise ValueError(f"Expected a run-local file, got {filename!r}.")
        if filename not in self.archive_files:
            self.archive_files.append(filename)

    def _archive(self):
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.archive_path.parent) as work:
            staging = Path(work) / self.archive_path.name
            with tarfile.open(staging, "w:gz") as archive:
                for filename in self.archive_files:
                    archive.add(self.path / filename, arcname=filename)
            staging.replace(self.archive_path)

    def __exit__(self, exc_type, exc, traceback):
        status = "succeeded"
        if exc is not None:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.logger.error("Run %s: %s", status, exc, exc_info=(exc_type, exc, traceback))
        ended = utc_now()
        exit_codes = {"succeeded": 0, "failed": 1, "interrupted": 130}
        summary = {
            **self.summary, "status": status, "ended_at": ended.isoformat(),
            "duration_seconds": (ended - self.started).total_seconds(),
            "exit_code": exit_codes[status],
        }
        if exc is not None:
            summary["error"] = str(exc)
        try:
            write_json(self.path / "summary.json", summary)
            self.logger.info("Run status %s | archive %s", status, self.archive_path)
            self._archive()
        except OSError as archive_error:
            self.logger.exception("Unable to finalize the log archive")
            summary["archive_error"] = str(archive_error)
            if exc is None:
                summary.update(status="failed", exit_code=1)
            try:
                write_json(self.path / "summary.json", summary)
            except OSError:
                self.logger.exception("Unable to update the local run summary")
            if exc is None:
                raise
        finally:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
        return False
