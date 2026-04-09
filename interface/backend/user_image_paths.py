from pathlib import Path
from typing import Optional


USER_IMAGES_DIR = Path(__file__).parent / "user_images"
USER_IMAGE_PREFIX = "user_images/"


def build_user_image_relpath(filename: str) -> str:
    return f"{USER_IMAGE_PREFIX}{filename}"


def resolve_user_image_path(stored_path: Optional[str]) -> Optional[str]:
    if not stored_path:
        return None
    if stored_path.startswith(USER_IMAGE_PREFIX):
        filename = stored_path[len(USER_IMAGE_PREFIX):]
        return str(USER_IMAGES_DIR / filename)
    return str((Path(__file__).parent / stored_path).resolve())
