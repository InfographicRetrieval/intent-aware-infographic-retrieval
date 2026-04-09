import os
import json
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from openai import OpenAI
from session_store import JsonUserSessionStore

# -------------------------
# Plain Chat: backend storage + 2-turn model memory
# -------------------------

PLAINCHAT_SESSIONS_DIR = os.environ.get(
    "PLAINCHAT_SESSIONS_DIR",
    os.path.join(os.path.dirname(__file__), "data", "plainchat", "users"),
)
PLAINCHAT_API_KEY = os.environ.get(
    "PLAINCHAT_API_KEY",
    os.environ.get("OPENAI_API_KEY", os.environ.get("CLOSEAI_API_KEY", "")),
)
PLAINCHAT_BASE_URL = os.environ.get(
    "PLAINCHAT_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", os.environ.get("CLOSEAI_BASE_URL", "https://api.openai.com/v1")),
)

DEFAULT_MODEL = os.environ.get("PLAINCHAT_DEFAULT_MODEL", "gpt-5.4")

router = APIRouter(prefix="/api/plainchat", tags=["plainchat"])

plainchat_session_store = JsonUserSessionStore(PLAINCHAT_SESSIONS_DIR)

def _now_iso() -> str:
    return datetime.now().isoformat()


def _with_user_sessions(username: str, write: bool = False):
    return plainchat_session_store.with_user_sessions(username, write=write)


def _ensure_session(user_sessions: Dict[str, Dict[str, Any]], session_id: str) -> Dict[str, Any]:
    sessions = user_sessions
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "name": "New Chat",
            "createdAt": _now_iso(),
            "lastMessageAt": _now_iso(),
            "messages": [],  # full history for display
        }
    return sessions[session_id]

def _list_sessions(user_sessions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for _, s in user_sessions.items():
        out.append({
            "id": s["id"],
            "name": s.get("name", "Chat"),
            "createdAt": s.get("createdAt", _now_iso()),
            "lastMessageAt": s.get("lastMessageAt", _now_iso()),
            "messageCount": len(s.get("messages", [])),
        })
    # sort by lastMessageAt desc
    out.sort(key=lambda x: x.get("lastMessageAt",""), reverse=True)
    return out

def _model_messages_from_history(
    full_history: List[Dict[str, Any]],
    user_text: str,
    user_image_data_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Model sees only last 2 turns (up to 4 messages) + current user message.

    Supports optional image on user messages using OpenAI vision message format.
    """
    # Keep only last 4 messages among user/assistant
    hist = [m for m in full_history if m.get("role") in ("user", "assistant")]
    hist = hist[-4:]

    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant. Reply with plain text only."}
    ]

    for m in hist:
        role = m.get("role")
        text = m.get("text", "")
        if role == "user" and m.get("image_data_url"):
            # Vision user message
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": m["image_data_url"]}},
                    ],
                }
            )
        else:
            msgs.append({"role": role, "content": text})

    # Current user message
    if user_image_data_url:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": user_image_data_url}},
                ],
            }
        )
    else:
        msgs.append({"role": "user", "content": user_text})

    return msgs


def _file_to_data_url(file: UploadFile) -> str:
    """Convert an uploaded image to a data URL for OpenAI vision input."""
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="user_image must be an image")
    raw = file.file.read()
    if raw is None:
        raise HTTPException(status_code=400, detail="Empty user_image")
    # Safety guard: limit to 8MB
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="user_image too large (max 8MB)")
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{content_type};base64,{b64}"

def _get_client() -> OpenAI:
    if not PLAINCHAT_API_KEY:
        raise RuntimeError("Missing PLAINCHAT_API_KEY or OPENAI_API_KEY env var")
    return OpenAI(api_key=PLAINCHAT_API_KEY, base_url=PLAINCHAT_BASE_URL)

class NewSessionReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)

class SendReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    session_id: str = Field(..., min_length=1, max_length=64)
    user_text: str = Field(..., min_length=1)
    model_name: Optional[str] = None

@router.get("/sessions")
def list_sessions(username: str):
    with _with_user_sessions(username, write=False) as user_sessions:
        return {"sessions": _list_sessions(user_sessions)}

@router.post("/new_session")
def new_session(req: NewSessionReq):
    with _with_user_sessions(req.username, write=True) as user_sessions:
        session_id = str(int(datetime.now().timestamp() * 1000))
        s = _ensure_session(user_sessions, session_id)
        s["name"] = "New Chat"
        s["createdAt"] = _now_iso()
        s["lastMessageAt"] = _now_iso()
        s["messages"] = []
        return {"session_id": session_id}

@router.get("/history")
def get_history(username: str, session_id: str):
    with _with_user_sessions(username, write=False) as user_sessions:
        s = _ensure_session(user_sessions, session_id)
        return {"history": s.get("messages", [])}

@router.post("/send")
def send(req: SendReq):
    with _with_user_sessions(req.username, write=True) as user_sessions:
        s = _ensure_session(user_sessions, req.session_id)

        # Append user message to full history (for display)
        user_msg = {
            "id": str(int(datetime.now().timestamp() * 1000)),
            "role": "user",
            "text": req.user_text,
            "timestamp": _now_iso(),
        }
        s["messages"].append(user_msg)

        # Call model with only last 2 turns
        model_name = (req.model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        messages_for_model = _model_messages_from_history(s["messages"][:-1], req.user_text)

        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages_for_model,
            )
            assistant_text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # rollback the user message? keep it for audit/debug. Up to you.
            raise HTTPException(status_code=500, detail=f"Model call failed: {e}")

        asst_msg = {
            "id": str(int(datetime.now().timestamp() * 1000) + 1),
            "role": "assistant",
            "text": assistant_text,
            "timestamp": _now_iso(),
        }
        s["messages"].append(asst_msg)

        # Update session metadata
        s["lastMessageAt"] = _now_iso()
        # Rename session based on first user message
        if s.get("name") in ("New Chat", "Chat", "", None):
            # pick first user message text
            first_user = next((m for m in s["messages"] if m.get("role") == "user"), None)
            if first_user and first_user.get("text"):
                s["name"] = (first_user["text"][:20] + "...") if len(first_user["text"]) > 20 else first_user["text"]

        return {"session_id": req.session_id, "output": assistant_text, "history": s["messages"]}


@router.post("/send_form")
def send_form(
    username: str = Form(...),
    session_id: str = Form(...),
    user_text: str = Form(...),
    model_name: Optional[str] = Form(None),
    user_image: Optional[UploadFile] = File(None),
):
    """Send a message with an optional image upload.

    Frontend should use multipart/form-data and post to /api/plainchat/send_form.
    """
    with _with_user_sessions(username, write=True) as user_sessions:
        s = _ensure_session(user_sessions, session_id)

        image_data_url: Optional[str] = None
        if user_image is not None:
            image_data_url = _file_to_data_url(user_image)

        # Append user message to full history (for display)
        user_msg: Dict[str, Any] = {
            "id": str(int(datetime.now().timestamp() * 1000)),
            "role": "user",
            "text": user_text,
            "timestamp": _now_iso(),
        }
        if image_data_url:
            user_msg["image_data_url"] = image_data_url
            user_msg["image_mime"] = user_image.content_type

        s["messages"].append(user_msg)

        # Call model with only last 2 turns (+ current image if provided)
        model_name_final = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        messages_for_model = _model_messages_from_history(
            s["messages"][:-1],
            user_text,
            user_image_data_url=image_data_url,
        )

        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model_name_final,
                messages=messages_for_model,
            )
            assistant_text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model call failed: {e}")

        asst_msg = {
            "id": str(int(datetime.now().timestamp() * 1000) + 1),
            "role": "assistant",
            "text": assistant_text,
            "timestamp": _now_iso(),
        }
        s["messages"].append(asst_msg)

        # Update session metadata
        s["lastMessageAt"] = _now_iso()
        if s.get("name") in ("New Chat", "Chat", "", None):
            first_user = next((m for m in s["messages"] if m.get("role") == "user"), None)
            if first_user and first_user.get("text"):
                s["name"] = (first_user["text"][:20] + "...") if len(first_user["text"]) > 20 else first_user["text"]

        return {"session_id": session_id, "output": assistant_text, "history": s["messages"]}
