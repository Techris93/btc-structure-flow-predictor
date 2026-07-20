from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import threading


logger = logging.getLogger("btc_predictor.persistence")


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
                if os.path.exists(tmp):
                    os.unlink(tmp)


def runtime_dir():
    """Prefer BTC_DATA_DIR, but fall back when the mount is not writable yet."""
    candidates = []
    configured = os.getenv("BTC_DATA_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path("/var/data"), Path("work/runtime"), Path(tempfile.gettempdir()) / "btc-structure-flow-predictor"])
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            if configured and path != Path(configured):
                logger.warning("BTC_DATA_DIR=%s is not writable; using %s", configured, path)
            return path
        except OSError as exc:
            logger.warning("Runtime directory unavailable at %s: %s", path, exc)
    raise RuntimeError("No writable runtime directory available")
