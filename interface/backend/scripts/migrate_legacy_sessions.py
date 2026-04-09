"""One-time migration tool for legacy main-chat sessions.

Usage:
    python interface/backend/scripts/migrate_legacy_sessions.py
"""

import json
import re
from pathlib import Path
from typing import Dict


BACKEND_DIR = Path(__file__).resolve().parent.parent
LEGACY_SESSIONS_FILE = BACKEND_DIR / "chat_sessions.json"
SESSIONS_DIR = BACKEND_DIR / "data" / "sessions" / "users"


def _safe_username(username: str) -> str:
    if not username:
        return "anonymous"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", username)[:128] or "anonymous"


def _user_sessions_file(username: str) -> Path:
    return SESSIONS_DIR / f"{_safe_username(username)}.json"


def _normalize_user_sessions(raw: Dict) -> Dict[str, Dict]:
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, Dict] = {}
    for session_id, session_data in raw.items():
        if isinstance(session_data, list):
            normalized[session_id] = {
                "messages": session_data,
                "svg_placeholder_map": {},
                "last_reference_image": None,
            }
        elif isinstance(session_data, dict):
            sd = dict(session_data)
            sd.setdefault("messages", [])
            sd.setdefault("svg_placeholder_map", {})
            sd.setdefault("last_reference_image", None)
            normalized[session_id] = sd
    return normalized


def _load_user_sessions(username: str) -> Dict[str, Dict]:
    target = _user_sessions_file(username)
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as f:
            return _normalize_user_sessions(json.load(f))
    except Exception as e:
        print(f"[WARN] Failed loading existing user sessions for {username}: {e}")
        return {}


def _save_user_sessions(username: str, user_sessions: Dict[str, Dict]) -> None:
    target = _user_sessions_file(username)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(user_sessions, f, ensure_ascii=False, indent=2)


def migrate() -> int:
    if not LEGACY_SESSIONS_FILE.exists():
        print(f"[SKIP] Legacy file not found: {LEGACY_SESSIONS_FILE}")
        return 0

    with LEGACY_SESSIONS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "users" in data:
        users_data = data.get("users", {})
    elif isinstance(data, dict):
        users_data = {"admin": data}
    else:
        users_data = {}

    migrated_users = 0
    for username, user_sessions in users_data.items():
        existing = _load_user_sessions(username)
        existing.update(_normalize_user_sessions(user_sessions if isinstance(user_sessions, dict) else {}))
        _save_user_sessions(username, existing)
        migrated_users += 1

    print(f"[DONE] Migrated legacy sessions for {migrated_users} user(s) into: {SESSIONS_DIR}")
    return migrated_users


if __name__ == "__main__":
    migrate()
