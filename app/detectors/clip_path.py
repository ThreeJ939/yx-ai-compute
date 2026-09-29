"""Ensure local CLIP ViT-B-32 weights are visible to Ultralytics / openai-clip."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CLIP_FILENAME = "ViT-B-32.pt"


def resolve_clip_file(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"CLIP weights not found: {p}")
    return p


def _openai_clip_cache_dir() -> Path:
    return Path.home() / ".cache" / "clip"


def _ultralytics_clip_cache_dir() -> Path | None:
    try:
        from ultralytics.utils import WEIGHTS_DIR

        return Path(WEIGHTS_DIR) / "clip"
    except Exception:
        return None


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if dest.resolve() == src.resolve():
                return
            # Same size shortcut: treat as already provisioned
            if dest.is_file() and dest.stat().st_size == src.stat().st_size:
                logger.debug("CLIP cache already present: %s", dest)
                return
        except OSError:
            pass
        dest.unlink()

    try:
        os.link(src, dest)
        logger.info("Hard-linked CLIP weights %s -> %s", src, dest)
        return
    except OSError:
        pass

    try:
        os.symlink(src, dest)
        logger.info("Symlinked CLIP weights %s -> %s", src, dest)
        return
    except OSError:
        pass

    shutil.copy2(src, dest)
    logger.info("Copied CLIP weights %s -> %s", src, dest)


def ensure_clip_caches(clip_weights: str | Path | None) -> Path | None:
    """
    Place configured CLIP .pt into locations Ultralytics / openai-clip look up.

    - Newer Ultralytics: ``WEIGHTS_DIR/clip/ViT-B-32.pt`` (download_root)
    - openai-clip default: ``~/.cache/clip/ViT-B-32.pt``

    Returns resolved source path, or None when clip_weights is empty.
    """
    if clip_weights is None:
        return None
    text = str(clip_weights).strip()
    if not text:
        return None

    src = resolve_clip_file(text)
    targets: list[Path] = [_openai_clip_cache_dir() / CLIP_FILENAME]
    ultra = _ultralytics_clip_cache_dir()
    if ultra is not None:
        targets.append(ultra / CLIP_FILENAME)

    for dest in targets:
        try:
            _link_or_copy(src, dest)
        except OSError:
            logger.exception("Failed to provision CLIP cache at %s", dest)
            raise

    # Hint for libraries that honor CLIP download root env (best-effort)
    os.environ.setdefault("CLIP_DOWNLOAD_ROOT", str(src.parent))
    return src
