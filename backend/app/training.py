"""Run training scripts in the background so they can be started from the web app."""
from __future__ import annotations

import subprocess
import sys
import threading
import time

from signvision.config import BACKEND_DIR, REPORTS_DIR

SCRIPTS = {"static": ["scripts.train_static"], "dynamic": ["scripts.train_dynamic"],
           "sentence": ["scripts.train_dynamic", "--kind", "sentence"]}


class TrainingManager:
    def __init__(self, on_finished):
        self._on_finished = on_finished
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def log_path(self, kind: str):
        return REPORTS_DIR / kind / "train.log"

    def start(self, kind: str) -> dict:
        with self._lock:
            job = self.jobs.get(kind)
            if job and job["state"] == "running":
                return job
            log = self.log_path(kind)
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", *SCRIPTS[kind]], cwd=BACKEND_DIR,
                stdout=fh, stderr=subprocess.STDOUT,
            )
            job = {"kind": kind, "state": "running", "started": time.time(), "returncode": None}
            self.jobs[kind] = job
            threading.Thread(target=self._wait, args=(kind, proc, fh), daemon=True).start()
            return job

    def _wait(self, kind, proc, fh):
        code = proc.wait()
        fh.close()
        with self._lock:
            self.jobs[kind].update(state="done" if code == 0 else "failed", returncode=code)
        if code == 0:
            self._on_finished()

    def status(self, kind: str) -> dict:
        job = dict(self.jobs.get(kind) or {"kind": kind, "state": "idle"})
        log = self.log_path(kind)
        job["log"] = log.read_text(encoding="utf-8", errors="replace")[-6000:] if log.exists() else ""
        return job
