"""Lifetime writer lease for the local JSONL store.

The lock file stays in place: unlinking it would allow a new inode to be locked
while an older process still owns the original. Kernel locks release on exit.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
from threading import Lock

_registry_lock = Lock()
_owned: set[Path] = set()


class WriterConflictError(RuntimeError):
    pass


class WriterLock:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir.resolve() / "writer.lock"
        self._file = None
        self._pid = os.getpid()
        with _registry_lock:
            if self.path in _owned:
                raise self._conflict()
            fh = self.path.open("a+b")
            try:
                if os.name == "nt":  # pragma: no cover - Windows
                    import msvcrt
                    if self.path.stat().st_size == 0:
                        fh.write(b"\0")
                        fh.flush()
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                fh.close()
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise self._conflict() from exc
                raise RuntimeError(f"Cannot acquire writer lease for {self.path.parent}: {exc}") from exc
            self._file = fh
            _owned.add(self.path)

    def _conflict(self) -> WriterConflictError:
        return WriterConflictError(
            f"Memory store {self.path.parent} already has an active writer. "
            "Use one shared HTTP server (memory-os-v4 --transport http) for multiple clients, "
            "or close the existing engine before opening another. Do not delete writer.lock."
        )

    def assert_owner(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("An inherited memory engine cannot be used after fork; connect to the shared HTTP server")

    def close(self) -> None:
        with _registry_lock:
            if self._file is None:
                return
            fh, self._file = self._file, None
            try:
                if os.getpid() != self._pid:
                    # A fork inherits the open file description. Explicitly
                    # unlocking it would also release the parent's lease.
                    return
                if os.name == "nt":  # pragma: no cover - Windows
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
                _owned.discard(self.path)
