from pathlib import Path

import pytest

from app.detectors.clip_path import CLIP_FILENAME, ensure_clip_caches, resolve_clip_file


def test_resolve_clip_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_clip_file(tmp_path / "missing.pt")


def test_ensure_clip_caches_empty():
    assert ensure_clip_caches("") is None
    assert ensure_clip_caches(None) is None


def test_ensure_clip_caches_provisions_openai_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "ViT-B-32.pt"
    src.write_bytes(b"fake-clip-weights")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Avoid depending on ultralytics in unit test
    monkeypatch.setattr(
        "app.detectors.clip_path._ultralytics_clip_cache_dir",
        lambda: tmp_path / "ultra" / "clip",
    )

    resolved = ensure_clip_caches(src)
    assert resolved == src.resolve()

    openai_dest = fake_home / ".cache" / "clip" / CLIP_FILENAME
    ultra_dest = tmp_path / "ultra" / "clip" / CLIP_FILENAME
    assert openai_dest.is_file()
    assert ultra_dest.is_file()
    assert openai_dest.read_bytes() == b"fake-clip-weights"
    assert ultra_dest.read_bytes() == b"fake-clip-weights"
