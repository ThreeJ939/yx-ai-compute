"""Tests for multi-backend registry routing and UltralyticsYoloDetector."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from app.config import Settings
from app.detectors.base import DetectBatchResult, RawDetection
from app.detectors.ultralytics_yolo import UltralyticsYoloDetector, _boxes_to_detections
from app.enhancers.crop_util import (
    crop_for_behavior,
    filter_fire_smoke,
    multi_union,
    single_expand,
)
from app.minio_client import MinioImageStore
from app.orchestrator import Orchestrator
from app.registry import collect_prompts, get_spec, plan_request


def _blank(w: int = 200, h: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(label: str, x: float, y: float, w: float, h: float) -> RawDetection:
    return RawDetection(
        object_type=label,
        label=label,
        confidence=0.9,
        bbox=[x, y, w, h],
        prompt=label,
    )


class _FakeDetector:
    def __init__(self, label: str, detections: list[RawDetection] | None = None):
        self.label = label
        self.calls = 0
        self.last_conf: float | None = None
        self.last_iou: float | None = None
        self._detections = detections

    def detect(self, image_bgr, prompts, conf, iou) -> DetectBatchResult:
        self.calls += 1
        self.last_conf = conf
        self.last_iou = iou
        if self._detections is not None:
            return DetectBatchResult(detections=list(self._detections))
        return DetectBatchResult(
            detections=[_det(self.label, 1, 2, 10, 20)]
        )


def test_fire_smoke_specs_backend():
    det = get_spec("DET_FIRE_SMOKE")
    beh = get_spec("BEH_FIRE_SMOKE")
    assert det is not None and det.backend == "fire_smoke" and det.stage == "detect"
    assert beh is not None and beh.backend == "fire_smoke" and beh.stage == "behavior"
    assert get_spec("DET_PERSON").backend == "world"


def test_collect_prompts_skips_non_world():
    prompts = collect_prompts(["DET_PERSON", "DET_FIRE_SMOKE", "BEH_FIRE_SMOKE"])
    assert prompts == ["person"]
    assert collect_prompts(["DET_FIRE_SMOKE"]) == []


def test_plan_request_by_backend():
    plan = plan_request(["DET_PERSON", "DET_FIRE_SMOKE", "BEH_FIRE_SMOKE", "NOPE"])
    assert plan.unknown == ["NOPE"]
    assert set(plan.by_backend.keys()) == {"world", "fire_smoke"}
    assert plan.by_backend["world"] == ["DET_PERSON"]
    assert plan.by_backend["fire_smoke"] == ["DET_FIRE_SMOKE", "BEH_FIRE_SMOKE"]
    assert plan.prompts == ["person"]


def test_crop_fire_smoke_expand_and_union():
    img = _blank(400, 300)
    dets = [_det("fire", 10, 20, 40, 80), _det("smoke", 200, 50, 50, 90), _det("person", 0, 0, 5, 5)]

    singles = single_expand(img, dets, filter_fire_smoke, padding=0.0)
    assert len(singles) == 2
    assert singles[0].det_indices == [0]
    assert singles[1].det_indices == [1]

    union = multi_union(img, dets, filter_fire_smoke, padding=0.0)
    assert len(union) == 1
    assert set(union[0].det_indices) == {0, 1}

    via = crop_for_behavior(img, dets, "single_fire_smoke_expand", 0.0)
    assert len(via) == 2
    via_u = crop_for_behavior(img, dets, "multi_fire_smoke_union", 0.0)
    assert len(via_u) == 1


def test_orchestrator_merges_backends():
    settings = Settings(llm_enabled=False, kafka_enabled=False)
    minio = MagicMock(spec=MinioImageStore)
    world = _FakeDetector("person")
    fire = _FakeDetector("fire")
    orch = Orchestrator(
        settings,
        minio,
        detector_registry={"world": world, "fire_smoke": fire},
    )
    img = _blank()
    plan = plan_request(["DET_PERSON", "DET_FIRE_SMOKE"])
    dets = orch._infer_plan_sync(img, plan, conf=0.2, iou=0.45)
    assert world.calls == 1
    assert fire.calls == 1
    assert {d.label for d in dets} == {"person", "fire"}
    orch.close()


def test_orchestrator_skips_missing_backend():
    settings = Settings(llm_enabled=False)
    orch = Orchestrator(
        settings,
        MagicMock(spec=MinioImageStore),
        detector_registry={"world": _FakeDetector("person")},
    )
    plan = plan_request(["DET_FIRE_SMOKE"])
    dets = orch._infer_plan_sync(_blank(), plan, 0.2, 0.45)
    assert dets == []
    orch.close()


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)

    def __len__(self) -> int:
        return int(self.xyxy.numpy().shape[0])


def test_boxes_to_detections_from_fake_result():
    boxes = _FakeBoxes(
        np.array([[10.0, 20.0, 40.0, 60.0]], dtype=np.float32),
        np.array([0.88], dtype=np.float32),
        np.array([0], dtype=np.int32),
    )
    result = SimpleNamespace(names={0: "Fire"}, boxes=boxes)
    batch = _boxes_to_detections([result])
    assert len(batch.detections) == 1
    assert batch.detections[0].label == "Fire"
    assert batch.detections[0].bbox == [10.0, 20.0, 30.0, 40.0]


def test_orchestrator_fire_smoke_uses_backend_conf_iou():
    settings = Settings(
        llm_enabled=False,
        kafka_enabled=False,
        yolo_conf=0.15,
        yolo_iou=0.45,
        fire_smoke_conf=0.3,
        fire_smoke_iou=0.55,
    )
    world = _FakeDetector("person")
    fire = _FakeDetector("fire")
    orch = Orchestrator(
        settings,
        MagicMock(spec=MinioImageStore),
        detector_registry={"world": world, "fire_smoke": fire},
    )
    plan = plan_request(["DET_PERSON", "DET_FIRE_SMOKE"])
    orch._infer_plan_sync(_blank(), plan, conf=0.15, iou=0.45)
    assert world.last_conf == 0.15
    assert world.last_iou == 0.45
    assert fire.last_conf == 0.3
    assert fire.last_iou == 0.55
    orch.close()


def test_apply_algo_conf_filter_by_prompt():
    from app.orchestrator import _apply_algo_conf_filter
    from app.registry import ALGORITHM_SPECS, AlgorithmSpec

    # Temporarily override DET_SHIP threshold via monkeypatch of get_spec path:
    # call filter with a synthetic codes list by patching get_spec locally.
    ship_high = AlgorithmSpec(
        code="DET_SHIP",
        name="船舶检测",
        stage="detect",
        backend="world",
        prompts=("boat", "vessel"),
        conf_threshold=0.5,
    )
    person_spec = get_spec("DET_PERSON")
    assert person_spec is not None

    dets = [
        RawDetection("person", "person", 0.2, [1, 2, 3, 4], prompt="person"),
        RawDetection("boat", "boat", 0.4, [1, 2, 3, 4], prompt="boat"),
        RawDetection("boat", "boat", 0.7, [1, 2, 3, 4], prompt="boat"),
    ]

    with patch("app.orchestrator.get_spec") as mock_get:
        def _get(code: str):
            if code == "DET_SHIP":
                return ship_high
            return ALGORITHM_SPECS.get(code)

        mock_get.side_effect = _get
        filtered = _apply_algo_conf_filter(dets, ["DET_PERSON", "DET_SHIP"])

    labels_conf = {(d.label, d.confidence) for d in filtered}
    assert ("person", 0.2) in labels_conf  # DET_PERSON threshold 0 → keep
    assert ("boat", 0.4) not in labels_conf  # below 0.5
    assert ("boat", 0.7) in labels_conf


def test_apply_algo_conf_filter_empty_prompt_backend():
    from app.orchestrator import _apply_algo_conf_filter
    from app.registry import AlgorithmSpec

    fire_spec = AlgorithmSpec(
        code="DET_FIRE_SMOKE",
        name="火焰烟雾检测",
        stage="detect",
        backend="fire_smoke",
        prompts=(),
        conf_threshold=0.6,
    )
    dets = [
        RawDetection("fire", "fire", 0.4, [1, 2, 3, 4]),
        RawDetection("smoke", "smoke", 0.8, [1, 2, 3, 4]),
    ]
    with patch("app.orchestrator.get_spec", return_value=fire_spec):
        filtered = _apply_algo_conf_filter(dets, ["DET_FIRE_SMOKE"])
    assert len(filtered) == 1
    assert filtered[0].label == "smoke"


def test_ultralytics_detector_full_image_mocked():
    img = _blank(100, 80)
    boxes = _FakeBoxes(
        np.array([[1.0, 2.0, 11.0, 22.0]], dtype=np.float32),
        np.array([0.77], dtype=np.float32),
        np.array([1], dtype=np.int32),
    )
    fake_result = SimpleNamespace(names={0: "Fire", 1: "Smoke"}, boxes=boxes)

    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    det = UltralyticsYoloDetector(
        weights="dummy.pt",
        device="cpu",
        tiled=False,
        name="fire_smoke",
    )
    with patch.object(det, "_ensure_model", return_value=fake_model):
        batch = det.detect(img, prompts=[], conf=0.25, iou=0.45)

    assert len(batch.detections) == 1
    assert batch.detections[0].label == "Smoke"
    fake_model.predict.assert_called_once()
