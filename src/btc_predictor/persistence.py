from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading


class JsonStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def read(self, default):
        with self.lock:
            try:
                return json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return default

    def write(self, value):
        with self.lock:
            fd, tmp = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(value, handle, default=str, separators=(",", ":"))
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)


def runtime_dir():
    path = Path(os.getenv("BTC_DATA_DIR", "work/runtime"))
    path.mkdir(parents=True, exist_ok=True)
    return path
