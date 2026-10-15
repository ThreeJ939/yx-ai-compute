import numpy as np
import pytest

from app.detectors.tiling import as_predict_result_list, build_tile_crops, iter_tiles


def test_iter_tiles_covers_corners():
    tiles = list(iter_tiles(1920, 1080, tile=640, overlap=0.2))
    assert tiles[0] == (0, 0)
    xs = {t[0] for t in tiles}
    ys = {t[1] for t in tiles}
    assert 0 in xs and max(xs) == 1920 - 640
    assert 0 in ys and max(ys) == 1080 - 640
    assert len(tiles) > 1


def test_iter_tiles_small_image_single_origin():
    tiles = list(iter_tiles(320, 240, tile=640, overlap=0.2))
    assert tiles == [(0, 0)]


def test_iter_tiles_overlap_increases_count():
    sparse = list(iter_tiles(1280, 720, tile=640, overlap=0.1))
    dense = list(iter_tiles(1280, 720, tile=640, overlap=0.4))
    assert len(dense) >= len(sparse)


def test_build_tile_crops_shapes_and_offsets():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[10, 20] = (1, 2, 3)
    crops = build_tile_crops(img, tile=640, overlap=0.2)
    origins = list(iter_tiles(1280, 720, tile=640, overlap=0.2))
    assert len(crops) == len(origins)
    for (x1, y1, crop), (ox, oy) in zip(crops, origins):
        assert (x1, y1) == (ox, oy)
        assert crop.shape == (640, 640, 3)
    # Pixel at (20,10) should appear in the tile that covers it
    assert crops[0][2][10, 20].tolist() == [1, 2, 3]


def test_as_predict_result_list_accepts_list_and_single():
    assert as_predict_result_list(["a", "b"], 2) == ["a", "b"]
    assert as_predict_result_list("one", 1) == ["one"]


def test_as_predict_result_list_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        as_predict_result_list(["a"], 2)
    with pytest.raises(ValueError, match="None"):
        as_predict_result_list(None, 1)
