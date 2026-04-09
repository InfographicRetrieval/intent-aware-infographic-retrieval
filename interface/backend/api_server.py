from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os, sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Generator
import json
from datetime import datetime
import asyncio
import time


# ===========================
# 路径配置
# ===========================
BACKEND_DIR = Path(__file__).resolve().parent
INTERFACE_DIR = BACKEND_DIR.parent
RELEASE_DIR = INTERFACE_DIR.parent
REPO_ROOT = RELEASE_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
HIERARCHY_FILE = Path(
    os.environ.get(
        "CHART_TYPES_HIERARCHY_FILE",
        str(REPO_ROOT / "data_processing" / "chart_types_hierarchy.json"),
    )
)
os.environ.setdefault("HF_HOME", "/mnt/share/xujing/hf")
# 添加项目路径
sys.path.append(str(BACKEND_DIR))
sys.path.append(str(REPO_ROOT / "retrieval_training"))
from retrieval_v3 import RetrievalV3

# 导入MLLM类
from mllm import MLLM
from session_store import JsonUserSessionStore
from user_image_paths import USER_IMAGES_DIR, build_user_image_relpath

# ===========================
# FastAPI应用配置
# ===========================
app = FastAPI(title="MLLM + Multimodal Retrieval API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from plain_chat_router import router as plainchat_router
app.include_router(plainchat_router)

# ===========================
# 全局配置
# ===========================
app_config = {
    'model_type': 'bge',
    'model_path': os.environ.get(
        "RETRIEVAL_CKPT",
        str(REPO_ROOT / "retrieval_training" / "output_4types" / "best_model.pt"),
    ),
    'base_dir': os.environ.get(
        "CHART_FEATURE_BASE_DIR",
        str(WORKSPACE_ROOT / "data" / "bge_caption_tuned"),
    ),
    'data_base_dir': os.environ.get('CHARTRETRIEVAL_DATA_ROOT', '/mnt/share/public/converted/converted'),
    'user_images_dir': str(INTERFACE_DIR / "backend/user_images"),
    'sessions_file': str(INTERFACE_DIR / "backend/chat_sessions.json")
}

# ===========================
# 检索默认参数
# ===========================
DEFAULT_CANDIDATE_TOPK = 20
DEFAULT_BASE_MODEL = "gpt-5.4"

# ===========================
# 初始化组件
# ===========================
retriever = RetrievalV3()

# 加载层次结构
chart_type_hierarchy = None
if HIERARCHY_FILE.exists():
    with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f:
        chart_type_hierarchy = json.load(f)
    print(f"Loaded chart type hierarchy: {len(chart_type_hierarchy['roots'])} root categories")
else:
    print(f"Warning: Hierarchy file not found at {HIERARCHY_FILE}")


def get_parent_chart_type(chart_type: str) -> Optional[str]:
    """Return parent type name for a leaf chart type via hierarchy flat_mapping."""
    if not chart_type_hierarchy or not chart_type:
        return None

    flat_mapping = chart_type_hierarchy.get('flat_mapping', {})
    mapping = flat_mapping.get(chart_type)
    if not mapping:
        return None

    full_path = [*(mapping.get('ancestors') or []), *(mapping.get('path') or [])]
    if len(full_path) < 2:
        return None

    for idx, name in enumerate(full_path):
        if name == chart_type and idx > 0:
            return full_path[idx - 1]

    return full_path[-2]

# 确保用户图片目录存在
os.makedirs(app_config['user_images_dir'], exist_ok=True)

# 静态文件服务 - 用于提供图片
app.mount("/static", StaticFiles(directory=app_config['data_base_dir']), name="static")

# ===========================
# 对话历史管理（主聊天：按用户分文件 + 跨进程文件锁）
# 唯一存储路径：interface/backend/data/sessions/users/{username}.json
# ===========================
SESSIONS_FILE = app_config['sessions_file']  # legacy path，仅供手动迁移脚本使用
SESSIONS_DIR = str(INTERFACE_DIR / "backend" / "data" / "sessions" / "users")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# 保留该变量仅用于兼容启动/关闭逻辑，不再作为真实存储源
chat_sessions: Dict[str, Dict[str, Dict]] = {}

# SVG 占位符管理（每个 session 独立的计数器）
import re
import copy
main_session_store = JsonUserSessionStore(SESSIONS_DIR, normalize_fn=lambda raw: _normalize_user_sessions(raw))


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
                "last_reference_images": [],
            }
        elif isinstance(session_data, dict):
            sd = dict(session_data)
            sd.setdefault("messages", [])
            sd.setdefault("svg_placeholder_map", {})
            sd.setdefault("last_reference_image", None)
            sd.setdefault("last_reference_images", [])
            if not sd["last_reference_images"] and sd.get("last_reference_image"):
                sd["last_reference_images"] = [sd["last_reference_image"]]
            normalized[session_id] = sd
    return normalized


def _load_user_sessions_unlocked(username: str) -> Dict[str, Dict]:
    user_file = main_session_store.user_sessions_file(username)
    if not os.path.exists(user_file):
        return {}
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            return _normalize_user_sessions(json.load(f))
    except Exception as e:
        print(f"Error loading user sessions for {username}: {e}")
        return {}


def _save_user_sessions_unlocked(username: str, user_sessions: Dict[str, Dict]) -> None:
    user_file = main_session_store.user_sessions_file(username)
    os.makedirs(os.path.dirname(user_file), exist_ok=True)
    temp_file = user_file + '.tmp'
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(user_sessions, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_file, user_file)


def _with_user_sessions(username: str, write: bool = False):
    return main_session_store.with_user_sessions(username, write=write)


def _ensure_session_data(user_sessions: Dict[str, Dict], session_id: str) -> Dict:
    if session_id not in user_sessions or isinstance(user_sessions[session_id], list):
        old_messages = user_sessions.get(session_id, [])
        if not isinstance(old_messages, list):
            old_messages = []
        user_sessions[session_id] = {
            "messages": old_messages,
            "svg_placeholder_map": {},
            "last_reference_image": None,
            "last_reference_images": [],
        }
    sd = user_sessions[session_id]
    sd.setdefault("messages", [])
    sd.setdefault("svg_placeholder_map", {})
    sd.setdefault("last_reference_image", None)
    sd.setdefault("last_reference_images", [])
    if not sd["last_reference_images"] and sd.get("last_reference_image"):
        sd["last_reference_images"] = [sd["last_reference_image"]]
    return sd


def _save_uploaded_user_image(upload: UploadFile, session_id: str) -> tuple[str, str]:
    import uuid

    file_extension = os.path.splitext(upload.filename or "image.png")[1]
    unique_filename = f"{session_id}_{uuid.uuid4().hex[:8]}{file_extension}"
    saved_image_path = str(USER_IMAGES_DIR / unique_filename)

    with open(saved_image_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return saved_image_path, build_user_image_relpath(unique_filename)


def get_or_create_session_data(username: str, session_id: str) -> Dict:
    """读取 session 快照（供模型调用读取 placeholder map）。"""
    with _with_user_sessions(username, write=False) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)
        return copy.deepcopy(session_data)


def _derive_session_display_name(session_data: Dict) -> str:
    explicit_name = str(session_data.get("name") or "").strip()
    if explicit_name:
        return explicit_name

    messages = session_data.get("messages", []) if isinstance(session_data, dict) else []
    if not messages:
        return "New Chat"

    first_content = messages[0].get("content", "New Chat")
    first_message = first_content if isinstance(first_content, str) else "New Chat"
    return first_message[:20] + "..." if len(first_message) > 20 else first_message


def _extract_last_reference_image(messages: List[Dict]) -> Optional[str]:
    latest_reference_images = _extract_latest_reference_images(messages)
    return latest_reference_images[0] if latest_reference_images else None


def _extract_latest_reference_images(messages: List[Dict]) -> List[str]:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        content = msg.get("content")
        if isinstance(content, list):
            assistant_images: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_reference" and item.get("image_path"):
                    assistant_images.append(str(item["image_path"]))
            if assistant_images:
                return assistant_images

        image_gallery = msg.get("image_gallery")
        if isinstance(image_gallery, list) and image_gallery:
            extracted_images: List[str] = []
            for item in image_gallery:
                if isinstance(item, dict):
                    chart_path = item.get("chart_path")
                    if chart_path:
                        extracted_images.append(str(chart_path))
                elif isinstance(item, str):
                    extracted_images.append(item)
            if extracted_images:
                return extracted_images

    return []


def compress_base64_in_text(text: str, placeholder_map: Dict[str, str]) -> tuple:
    """
    将文本中的 base64 图片数据替换为占位符

    Returns:
        (压缩后的文本, 更新后的占位符映射, 最大占位符编号)
    """
    if not text or 'data:image/' not in text:
        return text, placeholder_map, 0

    pattern = r"data:image/[^\"'>\s]{100,}"
    images_found = re.findall(pattern, text)

    if not images_found:
        return text, placeholder_map, 0

    max_num = 0
    for key in placeholder_map.keys():
        match = re.match(r'\[IMAGE_DATA_(\d+)\]', key)
        if match:
            max_num = max(max_num, int(match.group(1)))

    compressed_text = text
    for img_data in images_found:
        existing_placeholder = None
        for placeholder, data in placeholder_map.items():
            if data == img_data:
                existing_placeholder = placeholder
                break

        if existing_placeholder:
            compressed_text = compressed_text.replace(img_data, existing_placeholder, 1)
        else:
            max_num += 1
            placeholder = f"[IMAGE_DATA_{max_num}]"
            placeholder_map[placeholder] = img_data
            compressed_text = compressed_text.replace(img_data, placeholder, 1)

    return compressed_text, placeholder_map, max_num


def persist_placeholders_from_text(
    text: str,
    session_placeholder_map: Dict[str, str],
    source_placeholder_map: Optional[Dict[str, str]] = None,
) -> int:
    """把文本中出现的 [IMAGE_DATA_N]（若可解析）写入 session placeholder map。"""
    if not text or "[IMAGE_DATA_" not in text:
        return 0

    placeholders = set(re.findall(r'\[IMAGE_DATA_\d+\]', text))
    if not placeholders:
        return 0

    source_map = source_placeholder_map or {}
    added = 0
    for placeholder in placeholders:
        if placeholder in session_placeholder_map:
            continue
        value = source_map.get(placeholder)
        if value and value.startswith("data:image/"):
            session_placeholder_map[placeholder] = value
            added += 1
    return added


def prepare_assistant_text_for_storage(
    text: str,
    session_placeholder_map: Dict[str, str],
    source_placeholder_map: Optional[Dict[str, str]] = None,
) -> str:
    """统一处理 assistant 文本落盘：先回填 placeholder 映射，再压缩 base64。"""
    persist_placeholders_from_text(
        text,
        session_placeholder_map,
        source_placeholder_map=source_placeholder_map,
    )
    compressed_text, _, _ = compress_base64_in_text(text, session_placeholder_map)
    return compressed_text


def save_sessions_to_file():
    """兼容占位：主聊天已改为请求内按用户落盘。"""
    return


def auto_save_sessions():
    """兼容占位：主聊天已改为请求内按用户落盘。"""
    return


def _parse_candidate_count(raw_value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    """
    将candidate数量参数转换为整数
    """
    if raw_value is None or raw_value == "":
        return default
    
    try:
        value = int(raw_value)
        return value if value >= 0 else default
    except (ValueError, TypeError):
        return default

def restore_placeholders_in_text(text: str, placeholder_map: Dict[str, str]) -> str:
    """将占位符恢复为 base64 图片数据"""
    if not text or not placeholder_map:
        return text
    
    restored_text = text
    for placeholder, img_data in placeholder_map.items():
        restored_text = restored_text.replace(placeholder, img_data)
    
    return restored_text


def trim_chat_history_messages(messages: List[Dict], max_turns: int = 2) -> List[Dict]:
    """
    Trim chat history to at most `max_turns` user turns, keeping all messages from the earliest
    retained user message to the end. This is intended for *model context only* (do not use
    it to truncate stored history).

    A "turn" here is anchored by a message with role == "user".
    """
    if not messages or max_turns <= 0:
        return []

    user_count = 0
    start_idx = 0

    # Walk backwards to find the earliest message index to keep
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_count += 1
            if user_count == max_turns:
                start_idx = i
                break
    else:
        # Fewer than max_turns user messages -> keep all
        start_idx = 0

    return messages[start_idx:]


def get_chat_history(username: str, session_id: str, for_model: bool = False, restore_placeholders: bool = True) -> List[Dict]:
    """
    获取对话历史
    
    Args:
        username: 用户名
        session_id: 会话ID
        for_model: 是否为模型使用。如果为True，会过滤掉检索到的所有图片，只保留选中的图片
        restore_placeholders: 是否将占位符恢复为 base64（前端显示用True，模型用False）
    """
    with _with_user_sessions(username, write=False) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)
        history = list(session_data.get("messages", []))
        placeholder_map = dict(session_data.get("svg_placeholder_map", {}))
    
    if not for_model:
        # 前端显示：恢复占位符
        if restore_placeholders and placeholder_map:
            restored_history = []
            for msg in history:
                restored_msg = msg.copy()
                if "content" in msg:
                    if isinstance(msg["content"], str):
                        restored_msg["content"] = restore_placeholders_in_text(msg["content"], placeholder_map)
                    elif isinstance(msg["content"], list):
                        # 新格式，处理每个 content item
                        restored_content = []
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                restored_item = item.copy()
                                restored_item["text"] = restore_placeholders_in_text(item["text"], placeholder_map)
                                restored_content.append(restored_item)
                            else:
                                restored_content.append(item)
                        restored_msg["content"] = restored_content
                restored_history.append(restored_msg)
            return restored_history
        return history
    
    # 模型历史，需要过滤
    # 仅为模型输入裁剪到最近两轮（不影响存储与前端展示）
    history = trim_chat_history_messages(history, max_turns=2)

    filtered_history = []
    for msg in history:
        if msg["role"] == "user":
            filtered_history.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            })
        elif msg["role"] == "assistant":
            filtered_msg = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"]
            }
            filtered_history.append(filtered_msg)
    
    return filtered_history

def add_to_chat_history(username: str, session_id: str, user_message: str, assistant_response: str, 
                       retrieval_query: str = None, image_gallery: List[str] = None, 
                       selected_gallery: List[str] = None, selection_mode: str = None,
                       candidate_count: Optional[int] = None,
                       source_placeholder_map: Optional[Dict[str, str]] = None):
    """添加助手回复到历史记录（用户消息已在第一阶段添加）"""
    with _with_user_sessions(username, write=True) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)

        messages = session_data["messages"]
        placeholder_map = session_data["svg_placeholder_map"]

        # 统一处理：先回填 placeholder 映射，再压缩 base64
        compressed_response = prepare_assistant_text_for_storage(
            assistant_response,
            placeholder_map,
            source_placeholder_map=source_placeholder_map,
        )

        # 构建助手回复消息内容
        if selected_gallery:
            assistant_content = [
                {
                    "type": "text",
                    "text": compressed_response
                }
            ]

            for image_path in selected_gallery:
                assistant_content.append({
                    "type": "image_reference",
                    "image_path": image_path
                })

            assistant_message = {
                "role": "assistant", 
                "content": assistant_content,
                "timestamp": datetime.now().isoformat()
            }
        else:
            assistant_message = {
                "role": "assistant", 
                "content": compressed_response,
                "timestamp": datetime.now().isoformat()
            }

        # 添加额外的元数据（用于前端显示）
        if retrieval_query:
            assistant_message["retrieval_query"] = retrieval_query
        if image_gallery:
            assistant_message["image_gallery"] = image_gallery
        if selection_mode:
            assistant_message["selection_mode"] = selection_mode
        if candidate_count is not None:
            assistant_message["candidate_count"] = candidate_count

        # 维护 session 级别的最后一张 reference image（用于短历史下的后续对话）
        # 优先使用 selected_gallery（用户最终选中的图）；否则退化到 image_gallery 的第一张
        try:
            if selected_gallery and isinstance(selected_gallery, list) and len(selected_gallery) > 0:
                session_data["last_reference_images"] = list(selected_gallery)
                session_data["last_reference_image"] = selected_gallery[0]
            elif image_gallery and isinstance(image_gallery, list) and len(image_gallery) > 0:
                extracted_images = []
                for item in image_gallery:
                    if isinstance(item, dict):
                        chart_path = item.get("chart_path")
                        if chart_path:
                            extracted_images.append(str(chart_path))
                    elif isinstance(item, str):
                        extracted_images.append(item)
                session_data["last_reference_images"] = extracted_images
                session_data["last_reference_image"] = extracted_images[0] if extracted_images else None
        except Exception:
            pass

        messages.append(assistant_message)


# ===========================
# 核心业务逻辑
# ===========================

def refine_retrieval_logic(
    username: str,
    session_id: str,
    previous_query: str,
    refinement_text: str,
    user_image_path: Optional[str] = None,
    model_name: str = DEFAULT_BASE_MODEL,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOPK
):
    """
    Stage 5: 根据用户反馈精炼检索查询
    
    Args:
        username: 用户名
        session_id: 会话ID
        previous_query: 上一次检索使用的查询
        refinement_text: 用户提供的精炼指令
        user_image_path: 用户上传图片路径
        model_name: 调用的多模态模型名称
        candidate_top_k: 返回的候选图片数量上限
    """
    user_image = [user_image_path] if user_image_path else None
    history = get_chat_history(username, session_id, for_model=True, restore_placeholders=False) if session_id else []

    session_data = get_or_create_session_data(username, session_id)
    mllm = MLLM(llm_backend=model_name, external_placeholder_map=session_data["svg_placeholder_map"])

    message = mllm.create_multimodal_message(
        refinement_text, 
        user_image, 
        stage=5, 
        history=history,
        previous_query=previous_query
    )
    
    output = mllm.send_message(message)
    
    # 处理新的返回格式
    if isinstance(output, dict) and "content" in output:
        token_usage = output.get('usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        output_text = output["content"]
    else:
        token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        output_text = output
    
    print(f"[Stage 5 Token Usage] input: {token_usage['prompt_tokens']}, output: {token_usage['completion_tokens']}, total: {token_usage['total_tokens']}")
    
    query = mllm.parse_retrieval_query(output_text)
    
    # 从 5-aspect query 中提取 suggested_chart_types
    if query:
        ct_str = query.get('chart_type', {}).get('query', '')
        suggested_chart_types = [t.strip() for t in ct_str.split(',') if t.strip()]
    else:
        suggested_chart_types = []
    
    print("Refined query:", query)
    print("Suggested chart types:", suggested_chart_types)
    
    if query:
        results = retriever.search(
            query,
            top_k=candidate_top_k
        )["results"]
        print("Refined results:", results)

        # 返回完整的图片信息，包括chart_type
        image_gallery = [{
            'chart_path': result['chart_path'],
            'chart_type': result.get('chart_type', 'unknown'),
            'chart_type_parent': get_parent_chart_type(result.get('chart_type', 'unknown'))
        } for result in results]

        print("Returning refined retrieval results for user selection")
        
        return {
            "retrieval_query": json.dumps(query, ensure_ascii=False),
            "output": None,
            "image_gallery": image_gallery,
            "candidate_count": len(image_gallery),
            "used_history": len(history) > 0,
            "stage": "image_selection",
            "needs_user_selection": True,
            "suggested_chart_types": suggested_chart_types,
            "token_usage": token_usage
        }
    else:
        return {
            "retrieval_query": None,
            "output": "I am sorry, I could not refine the search based on your input. Please try again.",
            "image_gallery": [],
            "candidate_count": 0,
            "used_history": len(history) > 0,
            "stage": "completed",
            "token_usage": token_usage
        }


def reretrieve_with_query_logic(
    retrieval_query: str,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOPK,
):
    try:
        query = json.loads(retrieval_query)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid retrieval_query JSON: {exc}") from exc

    if not isinstance(query, dict):
        raise HTTPException(status_code=400, detail="retrieval_query must be a JSON object")

    results = retriever.search(
        query,
        top_k=candidate_top_k
    )["results"]

    image_gallery = [{
        'chart_path': result['chart_path'],
        'chart_type': result.get('chart_type', 'unknown'),
        'chart_type_parent': get_parent_chart_type(result.get('chart_type', 'unknown'))
    } for result in results]

    return {
        "retrieval_query": json.dumps(query, ensure_ascii=False),
        "output": None,
        "image_gallery": image_gallery,
        "candidate_count": len(image_gallery),
        "used_history": False,
        "stage": "image_selection",
        "needs_user_selection": True,
    }

def chat_logic(username: str, user_text: str, user_image_path: Optional[str] = None, 
               session_id: Optional[str] = None, model_name: str = DEFAULT_BASE_MODEL,
               candidate_top_k: int = DEFAULT_CANDIDATE_TOPK):
    """
    第一阶段：检索相关图片并返回给用户选择
    
    Args:
        username: 用户名
        user_text: 用户输入的文本
        user_image_path: 用户上传的图片路径
        session_id: 会话ID
        model_name: 模型名称
        candidate_top_k: 候选图片数量
    """
    user_image = [user_image_path] if user_image_path else None
    history = get_chat_history(username, session_id, for_model=True, restore_placeholders=False) if session_id else []

    # 读取 session 级别的最近一组 reference images（用于短历史下补充 reference）
    last_ref_images: List[str] = []
    try:
        session_data_tmp = get_or_create_session_data(username, session_id)
        if isinstance(session_data_tmp, dict):
            stored_reference_images = session_data_tmp.get("last_reference_images")
            if isinstance(stored_reference_images, list):
                last_ref_images = [str(item) for item in stored_reference_images if item]
            if not last_ref_images:
                last_ref_images = _extract_latest_reference_images(session_data_tmp.get("messages", []))
            if not last_ref_images and session_data_tmp.get("last_reference_image"):
                last_ref_images = [str(session_data_tmp["last_reference_image"])]
    except Exception:
        last_ref_images = []

    # 检查历史中是否有 reference images
    has_reference_images = False
    if history:
        for msg in history:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_reference":
                        has_reference_images = True
                        break
                if has_reference_images:
                    break
    
    session_data = get_or_create_session_data(username, session_id)
    mllm = MLLM(llm_backend=model_name, external_placeholder_map=session_data["svg_placeholder_map"])
    
    # 根据是否有 reference images 选择 stage
    has_any_reference = has_reference_images or bool(last_ref_images)
    selected_stage = 4 if has_any_reference else 0

    # 若裁剪后的历史里没有 reference，则用 session 里缓存的最近一组 reference 图补充给模型
    reference_images = last_ref_images if (not has_reference_images and last_ref_images) else None

    message = mllm.create_multimodal_message(
        user_text,
        user_image,
        stage=selected_stage,
        detail="high",
        history=history,
        reference_images=reference_images
    )
    use_tools = True if has_any_reference else False
    print("use tools:", use_tools)
    print("stage:", selected_stage)
    
    output = mllm.send_message(message, use_tools=use_tools)
    total_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    if isinstance(output, dict) and output.get("type") == "tool_call":
        print("LLM called tool")
        print("response:", output["message"])
        message.append(output["message"])
        # 保留 token 使用量
        if "usage" in output:
            total_token_usage = output["usage"].copy()
        output = mllm.handle_tool_calls(message, output["tool_calls"], total_token_usage)
        output_text = output["content"]
        total_token_usage = output.get("usage", total_token_usage)
    elif isinstance(output, dict) and "content" in output:
        output_text = output["content"]
        total_token_usage = output.get("usage", total_token_usage)
    else:
        output_text = output
    
    print(f"[Stage 4 Token Usage] input: {total_token_usage['prompt_tokens']}, output: {total_token_usage['completion_tokens']}, total: {total_token_usage['total_tokens']}")
    
    query = mllm.parse_retrieval_query(output_text)
    
    # 从 5-aspect query 中提取 suggested_chart_types
    if query:
        ct_str = query.get('chart_type', {}).get('query', '')
        suggested_chart_types = [t.strip() for t in ct_str.split(',') if t.strip()]
    else:
        suggested_chart_types = []
    
    print("query:", query)
    print("Suggested chart types:", suggested_chart_types)
    
    if query:
        results = retriever.search(
            query,
            top_k=candidate_top_k
        )["results"]
        # print("results:", results)


        # 返回完整的图片信息，包括chart_type
        image_gallery = [{
            'chart_path': result['chart_path'],
            'chart_type': result.get('chart_type', 'unknown'),
            'chart_type_parent': get_parent_chart_type(result.get('chart_type', 'unknown'))
        } for result in results]

        return {
            "retrieval_query": json.dumps(query, ensure_ascii=False),
            "output": None,
            "image_gallery": image_gallery,
            "candidate_count": len(image_gallery),
            "used_history": len(history) > 0 if session_id else False,
            "stage": "image_selection",
            "needs_user_selection": True,
            "suggested_chart_types": suggested_chart_types,
            "token_usage": total_token_usage,
            "source_placeholder_map": mllm.image_placeholder_map
        }
    else:
        output = mllm.parse_output(output_text, restore_placeholders=False)
        return {
            "retrieval_query": None,
            "output": output,
            "image_gallery": [],
            "candidate_count": 0,
            "used_history": len(history) > 0 if session_id else False,
            "stage": "completed",
            "token_usage": total_token_usage,
            "source_placeholder_map": mllm.image_placeholder_map
        }

def generate_final_response(username: str, user_text: str, image_gallery: List[str], user_image_path: Optional[str] = None, 
                          session_id: Optional[str] = None, selection_mode: str = "auto", 
                          model_name: str = DEFAULT_BASE_MODEL, retrieval_query: str = None,
                          candidate_count: Optional[int] = None):
    """
    第二阶段：根据用户选择的图片生成最终回答
    
    Args:
        username: 用户名
        user_text: 用户输入的文本
        image_gallery: 用户选择的图片路径列表
        user_image_path: 用户上传的图片路径
        session_id: 会话ID
        selection_mode: 选择模式，'auto'或'manual'
        model_name: 模型名称
        retrieval_query: 第一阶段的检索查询
        candidate_count: 第一阶段候选图片数量
    """
    user_image = [user_image_path] if user_image_path else None
    history = get_chat_history(username, session_id, for_model=True, restore_placeholders=False) if session_id else []
    
    session_data = get_or_create_session_data(username, session_id)
    mllm = MLLM(llm_backend=model_name, external_placeholder_map=session_data["svg_placeholder_map"])
    
    # 累计 token 使用量
    total_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # 如果是auto模式，先进行Stage 1选择
    if selection_mode == "auto":
        print("自动模式：进行AI选择")
        message = mllm.create_multimodal_message(user_text, user_image, image_gallery, 
                                                stage=1, history=history)
        output_stage_1 = mllm.send_message(message)
        print("output_stage_1:", output_stage_1)
        
        # 处理新的返回格式
        if isinstance(output_stage_1, dict) and "content" in output_stage_1:
            total_token_usage['prompt_tokens'] += output_stage_1.get('usage', {}).get('prompt_tokens', 0)
            total_token_usage['completion_tokens'] += output_stage_1.get('usage', {}).get('completion_tokens', 0)
            total_token_usage['total_tokens'] += output_stage_1.get('usage', {}).get('total_tokens', 0)
            output_stage_1_text = output_stage_1["content"]
        else:
            output_stage_1_text = output_stage_1
        
        select = mllm.parse_select(output_stage_1_text)
        
        # 处理选择结果：去重、过滤无效索引、保持顺序
        if select and isinstance(select, list):
            seen = set()
            selected_gallery = []
            for i in select:
                if isinstance(i, int) and 0 <= i < len(image_gallery) and i not in seen:
                    seen.add(i)
                    selected_gallery.append(image_gallery[i])
            print(f"AI selected {len(selected_gallery)} unique images from {len(select)} indices")
        else:
            # 如果解析失败，使用所有图片
            selected_gallery = image_gallery
            print(f"Failed to parse AI selection, using all {len(image_gallery)} images")
    else:
        print(f"手动模式：使用用户选择的 {len(image_gallery)} 张图片")
        selected_gallery = image_gallery
    
    # Stage 2: 生成最终回答
    message = mllm.create_multimodal_message(user_text, user_image, selected_gallery, 
                                            detail="high", stage=2, history=history)
    output_stage_2 = mllm.send_message(message)
    print("output_stage_2:", output_stage_2)
    
    # 处理新的返回格式
    if isinstance(output_stage_2, dict) and "content" in output_stage_2:
        total_token_usage['prompt_tokens'] += output_stage_2.get('usage', {}).get('prompt_tokens', 0)
        total_token_usage['completion_tokens'] += output_stage_2.get('usage', {}).get('completion_tokens', 0)
        total_token_usage['total_tokens'] += output_stage_2.get('usage', {}).get('total_tokens', 0)
        output_stage_2_text = output_stage_2["content"]
    else:
        output_stage_2_text = output_stage_2
    
    output = mllm.parse_output(output_stage_2_text, restore_placeholders=False)
    print(output)
    
    print(f"[Stage 1+2 Token Usage] input: {total_token_usage['prompt_tokens']}, output: {total_token_usage['completion_tokens']}, total: {total_token_usage['total_tokens']}")
    
    effective_candidate_count = candidate_count if candidate_count is not None else len(image_gallery)
    
    return {
        "output": output,
        "selected_gallery": selected_gallery,
        "selection_mode": selection_mode,
        "used_history": len(history) > 0,
        "stage": "completed",
        "retrieval_query": retrieval_query,
        "image_gallery": image_gallery,
        "candidate_count": effective_candidate_count,
        "token_usage": total_token_usage,
        "source_placeholder_map": mllm.image_placeholder_map
    }

def generate_direct_response(username: str, user_text: str, user_image_path: Optional[str] = None, 
                            session_id: Optional[str] = None, 
                            model_name: str = DEFAULT_BASE_MODEL):
    """
    直接回答模式：不使用参考图片，直接生成回答
    
    Args:
        username: 用户名
        user_text: 用户输入的文本
        user_image_path: 用户上传的图片路径
        session_id: 会话ID
        model_name: 模型名称
    """
    user_image = [user_image_path] if user_image_path else None
    history = get_chat_history(username, session_id, for_model=True, restore_placeholders=False) if session_id else []
    
    session_data = get_or_create_session_data(username, session_id)
    mllm = MLLM(llm_backend=model_name, external_placeholder_map=session_data["svg_placeholder_map"])
    
    message = mllm.create_multimodal_message(user_text, user_image, stage=3, history=history)
    output = mllm.send_message(message)
    
    # 处理新的返回格式
    if isinstance(output, dict) and "content" in output:
        token_usage = output.get('usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        output_text = output["content"]
    else:
        token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        output_text = output
    
    print(f"[Stage 3 Token Usage] input: {token_usage['prompt_tokens']}, output: {token_usage['completion_tokens']}, total: {token_usage['total_tokens']}")
    
    return {
        "output": output_text,
        "selected_gallery": [],
        "selection_mode": "direct",
        "used_history": len(history) > 0,
        "stage": "completed",
        "token_usage": token_usage,
        "source_placeholder_map": mllm.image_placeholder_map
    }

# ===========================
# API端点
# ===========================

@app.get("/api/chart-types")
async def get_chart_types_endpoint():
    """获取chart type层次结构"""
    if not chart_type_hierarchy:
        raise HTTPException(status_code=500, detail="Chart type hierarchy not loaded")
    
    # 返回层次结构，前端可以用来构建树形UI
    return {
        "hierarchy": chart_type_hierarchy['roots'],
        "flat_mapping": chart_type_hierarchy['flat_mapping'],
        "statistics": chart_type_hierarchy['statistics'],
        "default_selection": [root['name'] for root in chart_type_hierarchy['roots']]  # 默认选择所有根节点
    }

@app.post("/api/chat")
async def chat_endpoint(
    username: str = Form(...),
    user_text: str = Form(...),
    session_id: str = Form(...),
    model_name: str = Form(DEFAULT_BASE_MODEL),
    user_image: Optional[UploadFile] = File(None)
):
    """第一阶段：检索相关图片并返回给用户选择"""
    user_image_path = None
    saved_user_image_path = None
    
    if user_image:
        user_image_path, saved_user_image_path = _save_uploaded_user_image(user_image, session_id)
    
    result = chat_logic(
        username,
        user_text,
        user_image_path,
        session_id,
        model_name
    )
    
    user_message = {
        "role": "user",
        "content": user_text,
        "timestamp": datetime.now().isoformat()
    }

    if saved_user_image_path:
        user_message["user_image_path"] = saved_user_image_path

    with _with_user_sessions(username, write=True) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)
        session_data["messages"].append(user_message)

        if result.get("stage") == "completed" and result.get("output"):
            placeholder_map = session_data["svg_placeholder_map"]
            compressed = prepare_assistant_text_for_storage(
                result["output"],
                placeholder_map,
                source_placeholder_map=result.get("source_placeholder_map"),
            )
            session_data["messages"].append({
                "role": "assistant",
                "content": compressed,
                "timestamp": datetime.now().isoformat()
            })
    
    result.pop("source_placeholder_map", None)
    return result

@app.post("/api/chat/refine")
async def refine_chat_endpoint(
    username: str = Form(...),
    session_id: str = Form(...),
    previous_query: str = Form(...),
    refinement_text: str = Form(...),
    model_name: str = Form(DEFAULT_BASE_MODEL),
    user_image: Optional[UploadFile] = File(None)
):
    """Endpoint for refining the retrieval query"""
    if not previous_query or not previous_query.strip():
        raise HTTPException(status_code=400, detail="previous_query is required for refinement")
    user_image_path = None
    saved_user_image_path = None
    
    if user_image:
        user_image_path, saved_user_image_path = _save_uploaded_user_image(user_image, session_id)

    user_message = {
        "role": "user",
        "content": refinement_text,
        "timestamp": datetime.now().isoformat()
    }
    if saved_user_image_path:
        user_message["user_image_path"] = saved_user_image_path

    with _with_user_sessions(username, write=True) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)
        session_data["messages"].append(user_message)

    result = refine_retrieval_logic(
        username=username,
        session_id=session_id,
        previous_query=previous_query,
        refinement_text=refinement_text,
        user_image_path=user_image_path,
        model_name=model_name
    )

    return result


@app.post("/api/chat/reretrieve")
async def reretrieve_chat_endpoint(
    username: str = Form(...),
    session_id: str = Form(...),
    retrieval_query: str = Form(...),
    candidate_count: Optional[str] = Form(None),
):
    """Re-run retrieval with a user-adjusted 5-aspect spec without adding chat history."""
    _ = username
    _ = session_id
    candidate_top_k = _parse_candidate_count(candidate_count, DEFAULT_CANDIDATE_TOPK) or DEFAULT_CANDIDATE_TOPK
    return reretrieve_with_query_logic(
        retrieval_query=retrieval_query,
        candidate_top_k=candidate_top_k,
    )

@app.post("/api/chat/finalize")
async def finalize_chat_endpoint(
    username: str = Form(...),
    user_text: str = Form(...),
    session_id: str = Form(...),
    selected_images: str = Form(None),
    selection_mode: str = Form("manual"),
    model_name: str = Form(DEFAULT_BASE_MODEL),
    retrieval_query: str = Form(None),
    user_image: Optional[UploadFile] = File(None),
    candidate_count: Optional[str] = Form(None)
):
    """第二阶段：根据用户选择生成最终回答"""
    user_image_path = None
    candidate_total = _parse_candidate_count(candidate_count)
    
    if user_image:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, user_image.filename or "uploaded_image")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(user_image.file, buffer)
        
        user_image_path = temp_file_path
    
    if selection_mode == "direct":
        result = generate_direct_response(
            username=username,
            user_text=user_text,
            user_image_path=user_image_path,
            session_id=session_id,
            model_name=model_name
        )
    else:
        if not selected_images:
            raise HTTPException(status_code=400, detail="selected_images is required for auto/manual mode")
            
        selected_gallery = json.loads(selected_images)
        if not isinstance(selected_gallery, list):
            raise HTTPException(status_code=400, detail="selected_images must be a list")
        
        if candidate_total is None:
            candidate_total = len(selected_gallery)
        
        result = generate_final_response(
            username=username,
            user_text=user_text,
            image_gallery=selected_gallery,
            user_image_path=user_image_path,
            session_id=session_id,
            selection_mode=selection_mode,
            model_name=model_name,
            retrieval_query=retrieval_query,
            candidate_count=candidate_total
        )
    
    add_to_chat_history(
        username=username,
        session_id=session_id, 
        user_message=user_text, 
        assistant_response=result["output"],
        retrieval_query=result.get("retrieval_query"),
        image_gallery=result.get("image_gallery"),
        selected_gallery=result.get("selected_gallery"),
        selection_mode=result.get("selection_mode"),
        candidate_count=result.get("candidate_count"),
        source_placeholder_map=result.get("source_placeholder_map")
    )

    result.pop("source_placeholder_map", None)
    return result

@app.get("/api/image/{image_path:path}")
async def get_image(image_path: str):
    """获取图片文件的API端点"""
    try:
        if image_path.startswith("user_images/"):
            filename = image_path.replace("user_images/", "")
            full_path = os.path.join(app_config['user_images_dir'], filename)
        else:
            full_path = os.path.join(app_config['data_base_dir'], image_path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(full_path)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving image {image_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions_endpoint(username: str):
    """获取所有会话列表的API端点"""
    sessions_list = []
    with _with_user_sessions(username, write=False) as user_sessions:
        for session_id, session_data in user_sessions.items():
            messages = session_data.get('messages', []) if isinstance(session_data, dict) else []
            if messages:
                last_message_time = messages[-1].get('timestamp', datetime.now().isoformat()) if messages else datetime.now().isoformat()

                sessions_list.append({
                    "id": session_id,
                    "name": _derive_session_display_name(session_data),
                    "createdAt": messages[0].get('timestamp', datetime.now().isoformat()) if messages else datetime.now().isoformat(),
                    "lastMessageAt": last_message_time,
                    "messageCount": len(messages)
                })

    sessions_list.sort(key=lambda x: x['lastMessageAt'], reverse=True)
    return {"sessions": sessions_list}


@app.post("/api/chat/history/{session_id}/copy")
async def copy_chat_session_endpoint(username: str, session_id: str):
    """Duplicate a session under a new id for the same user."""
    with _with_user_sessions(username, write=True) as user_sessions:
        if session_id not in user_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        source_session = _ensure_session_data(user_sessions, session_id)
        copied_session = copy.deepcopy(source_session)
        new_session_id = str(int(time.time() * 1000))
        while new_session_id in user_sessions:
            new_session_id = str(int(new_session_id) + 1)

        copied_session["name"] = f"{_derive_session_display_name(source_session)} (Copy)"
        user_sessions[new_session_id] = copied_session

    return {"session_id": new_session_id}


@app.delete("/api/chat/history/{session_id}/last-turn")
async def delete_last_turn_endpoint(username: str, session_id: str):
    """Delete the latest user turn and everything after it within a session."""
    with _with_user_sessions(username, write=True) as user_sessions:
        session_data = _ensure_session_data(user_sessions, session_id)
        messages = session_data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Session has no messages")

        last_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx < 0:
            raise HTTPException(status_code=400, detail="No user turn found to delete")

        del messages[last_user_idx:]
        latest_reference_images = _extract_latest_reference_images(messages)
        session_data["last_reference_images"] = latest_reference_images
        session_data["last_reference_image"] = latest_reference_images[0] if latest_reference_images else None

    return {"session_id": session_id, "remaining_messages": len(messages)}

@app.get("/api/chat/history/{session_id}")
async def get_chat_history_endpoint(username: str, session_id: str):
    try:
        # 1) 不 restore，避免返回巨大 base64（可选）
        history = get_chat_history(username, session_id, restore_placeholders=False)

        # 2) 把 placeholder_map 一并返回（以文件里的为准）
        with _with_user_sessions(username, write=False) as user_sessions:
            session_data = _ensure_session_data(user_sessions, session_id)
            placeholder_map = session_data.get("svg_placeholder_map", {})

        return {
            "session_id": session_id,
            "history": history,
            "svg_placeholder_map": placeholder_map
        }
    except Exception as e:
        print(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history/{session_id}")
async def clear_chat_history_endpoint(username: str, session_id: str):
    """清空对话历史的API端点"""
    try:
        with _with_user_sessions(username, write=True) as user_sessions:
            if session_id in user_sessions:
                del user_sessions[session_id]
        return {"message": f"Chat history for session {session_id} cleared"}
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "MLLM API is running"}

# ===========================
# 简易 Baseline 检索端点
# ===========================
@app.post("/api/baseline/search")
async def baseline_search_endpoint(
    query: str = Form(...),
    top_k: int = Form(10)
):
    """
    简易 Baseline 检索端点
    直接使用文本查询，返回最相似的图片列表
    """
    try:
        from retrieval_baseline import baseline_search
        
        results = baseline_search(query, top_k=top_k)
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        print(f"Baseline search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===========================
# 应用生命周期事件
# ===========================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行的初始化任务"""
    print("Starting MLLM API server...")
    print("Per-user session store ready")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的清理任务"""
    print("Shutting down MLLM API server...")
    print("Main chat sessions are persisted per request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
