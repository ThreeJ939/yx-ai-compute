from app.detectors.tiling import iter_tiles


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
