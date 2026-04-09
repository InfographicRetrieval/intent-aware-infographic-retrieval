import fcntl
import json
import os
import re
import threading
from typing import Any, Callable, Dict, Optional


SessionDict = Dict[str, Dict[str, Any]]
NormalizeFn = Callable[[Any], SessionDict]


def default_normalize_sessions(raw: Any) -> SessionDict:
    return raw if isinstance(raw, dict) else {}


class JsonUserSessionStore:
    def __init__(self, base_dir: str, normalize_fn: Optional[NormalizeFn] = None):
        self.base_dir = base_dir
        self.normalize_fn = normalize_fn or default_normalize_sessions
        self._user_locks: Dict[str, threading.Lock] = {}
        self._user_locks_guard = threading.Lock()

    @staticmethod
    def safe_username(username: str) -> str:
        if not username:
            return "anonymous"
        return re.sub(r"[^a-zA-Z0-9._-]", "_", username)[:128] or "anonymous"

    def user_sessions_file(self, username: str) -> str:
        return os.path.join(self.base_dir, f"{self.safe_username(username)}.json")

    def user_lock_file(self, username: str) -> str:
        return os.path.join(self.base_dir, f"{self.safe_username(username)}.lock")

    def _get_user_thread_lock(self, username: str) -> threading.Lock:
        key = self.safe_username(username)
        with self._user_locks_guard:
            if key not in self._user_locks:
                self._user_locks[key] = threading.Lock()
            return self._user_locks[key]

    def _load_user_sessions_unlocked(self, username: str) -> SessionDict:
        user_file = self.user_sessions_file(username)
        if not os.path.exists(user_file):
            return {}
        try:
            with open(user_file, "r", encoding="utf-8") as f:
                return self.normalize_fn(json.load(f))
        except Exception:
            return {}

    def _save_user_sessions_unlocked(self, username: str, user_sessions: SessionDict) -> None:
        user_file = self.user_sessions_file(username)
        os.makedirs(os.path.dirname(user_file), exist_ok=True)
        temp_file = user_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(user_sessions, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, user_file)

    def with_user_sessions(self, username: str, write: bool = False):
        store = self

        class _Ctx:
            def __init__(self):
                self.username = username
                self.write = write
                self.thread_lock = store._get_user_thread_lock(username)
                self.lock_fp = None
                self.user_sessions: SessionDict = {}

            def __enter__(self):
                self.thread_lock.acquire()
                lock_path = store.user_lock_file(self.username)
                os.makedirs(os.path.dirname(lock_path), exist_ok=True)
                self.lock_fp = open(lock_path, "a+", encoding="utf-8")
                fcntl.flock(self.lock_fp.fileno(), fcntl.LOCK_EX)
                self.user_sessions = store._load_user_sessions_unlocked(self.username)
                return self.user_sessions

            def __exit__(self, exc_type, exc, tb):
                try:
                    if self.write and exc_type is None:
                        store._save_user_sessions_unlocked(self.username, self.user_sessions)
                finally:
                    if self.lock_fp is not None:
                        fcntl.flock(self.lock_fp.fileno(), fcntl.LOCK_UN)
                        self.lock_fp.close()
                    self.thread_lock.release()

        return _Ctx()
