"""Unit tests for LLM behavior enhancer (mocked VLM client)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import Settings
from app.detectors.base import RawDetection
from app.enhancers.base import EnhanceRequest
from app.enhancers.crop_util import (
    crop_for_behavior,
    multi_person_union,
    single_person_expand,
)
from app.enhancers.llm_client import normalize_verdict, parse_vlm_json
from app.enhancers.llm_vlm import (
    LlmBehaviorEnhancer,
    PromptTemplate,
    apply_verdicts,
    load_llm_prompts,
)


def _person(x: float, y: float, w: float, h: float, conf: float = 0.9) -> RawDetection:
    return RawDetection(
        object_type="person",
        label="person",
        confidence=conf,
        bbox=[x, y, w, h],
        prompt="person",
    )


def _blank_image(w: int = 200, h: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_parse_vlm_json_plain_and_fenced():
    plain = parse_vlm_json('{"positive": true, "confidence": 0.88, "label": "fishing"}')
    assert plain["positive"] is True
    assert plain["confidence"] == 0.88

    fenced = parse_vlm_json(
        '这里是结果:\n```json\n{"positive": false, "confidence": 0.2, "label": "fighting"}\n```\n'
    )
    assert fenced["positive"] is False
    assert normalize_verdict(fenced)["label"] == "fighting"


def test_normalize_verdict_string_bool():
    v = normalize_verdict({"positive": "yes", "confidence": "0.7", "label": "fishing"})
    assert v["positive"] is True
    assert v["confidence"] == 0.7


def test_single_person_expand_and_union():
    img = _blank_image(400, 300)
    dets = [_person(10, 20, 40, 80), _person(200, 50, 50, 90)]

    singles = single_person_expand(img, dets, padding=0.0)
    assert len(singles) == 2
    assert singles[0].det_indices == [0]
    assert singles[1].det_indices == [1]

    union = multi_person_union(img, dets, padding=0.0)
    assert len(union) == 1
    assert set(union[0].det_indices) == {0, 1}
    x, y, w, h = union[0].bbox_xywh
    assert x == 10 and y == 20
    assert abs(w - (250 - 10)) < 1e-6
    assert abs(h - (140 - 20)) < 1e-6

    via_mode = crop_for_behavior(img, dets, "multi_person_union", 0.0)
    assert len(via_mode) == 1


def test_apply_verdicts_conservative():
    dets = [_person(0, 0, 10, 10, conf=0.9), _person(20, 0, 10, 10, conf=0.8)]
    from app.enhancers.crop_util import CropItem

    crops = [
        CropItem(image_bgr=_blank_image(10, 10), det_indices=[0], bbox_xywh=[0, 0, 10, 10]),
        CropItem(image_bgr=_blank_image(10, 10), det_indices=[1], bbox_xywh=[20, 0, 10, 10]),
    ]
    verdicts = [
        {"positive": True, "confidence": 0.9, "label": "fishing"},
        {"positive": False, "confidence": 0.1, "label": "fishing"},
    ]
    out = apply_verdicts(dets, crops, verdicts, conf_threshold=0.5)
    assert len(out) == 2  # never drop boxes
    assert out[0].behavior == "fishing"
    assert out[0].confidence == 0.9
    assert out[1].behavior is None
    assert out[1].confidence == 0.8


def test_load_prompts_yaml():
    root = Path(__file__).resolve().parents[1]
    prompts = load_llm_prompts(root / "config" / "llm_prompts.yaml")
    assert "BEH_FISHING" in prompts
    assert "BEH_FIGHTING" in prompts
    assert prompts["BEH_FIGHTING"].crop == "multi_person_union"
    assert "BEH_SMOKING" in prompts
    assert prompts["BEH_SMOKING"].crop == "single_person_expand"
    assert "BEH_FIRE_SMOKE" in prompts
    assert prompts["BEH_FIRE_SMOKE"].crop == "single_fire_smoke_expand"


class _FakeClient:
    def __init__(self, verdicts: list[dict] | Exception):
        self._verdicts = verdicts
        self.calls = 0

    def encode_crop_jpeg_b64(self, image_bgr, quality: int = 85) -> str:
        return "ZmFrZQ=="

    def call(self, crops_b64, system, user, max_workers: int = 1):
        self.calls += 1
        if isinstance(self._verdicts, Exception):
            raise self._verdicts
        return list(self._verdicts)


def test_enhancer_fishing_single_crop_positive():
    settings = Settings(llm_enabled=True, llm_conf_threshold=0.5)
    prompts = {
        "BEH_FISHING": PromptTemplate(
            system="sys",
            user="usr",
            crop="single_person_expand",
            crop_padding=0.0,
        )
    }
    client = _FakeClient(
        [{"positive": True, "confidence": 0.91, "label": "fishing"}]
    )
    enhancer = LlmBehaviorEnhancer(settings, client=client, prompts=prompts)
    img = _blank_image()
    dets = [_person(5, 5, 30, 40)]
    result = enhancer.enhance(
        EnhanceRequest(
            behavior="BEH_FISHING",
            image_bgr=img,
            detections=dets,
            algorithm_code="BEH_FISHING",
        )
    )
    assert result.applied is True
    assert result.detections[0].behavior == "fishing"
    assert client.calls == 1
    assert len(result.crops) == 1
    assert result.crops[0]["image_b64"]
    assert result.crops[0]["algorithm_code"] == "BEH_FISHING"
    assert result.crops[0]["positive"] is True
    assert result.crops[0]["label"] == "fishing"


def test_enhancer_fighting_union_crop():
    settings = Settings(llm_enabled=True, llm_conf_threshold=0.5)
    prompts = {
        "BEH_FIGHTING": PromptTemplate(
            system="sys",
            user="usr",
            crop="multi_person_union",
            crop_padding=0.0,
        )
    }
    client = _FakeClient(
        [{"positive": True, "confidence": 0.8, "label": "fighting"}]
    )
    enhancer = LlmBehaviorEnhancer(settings, client=client, prompts=prompts)
    img = _blank_image(300, 200)
    dets = [_person(10, 10, 40, 60), _person(80, 20, 40, 60)]
    result = enhancer.enhance(
        EnhanceRequest(
            behavior="BEH_FIGHTING",
            image_bgr=img,
            detections=dets,
            algorithm_code="BEH_FIGHTING",
        )
    )
    assert result.applied is True
    assert all(d.behavior == "fighting" for d in result.detections)
    assert client.calls == 1


def test_enhancer_degrades_on_client_error():
    settings = Settings(llm_enabled=True)
    prompts = {
        "BEH_FISHING": PromptTemplate(
            system="sys", user="usr", crop="single_person_expand", crop_padding=0.0
        )
    }
    client = _FakeClient(TimeoutError("vlm timeout"))
    enhancer = LlmBehaviorEnhancer(settings, client=client, prompts=prompts)
    dets = [_person(5, 5, 30, 40)]
    result = enhancer.enhance(
        EnhanceRequest(
            behavior="BEH_FISHING",
            image_bgr=_blank_image(),
            detections=dets,
            algorithm_code="BEH_FISHING",
        )
    )
    assert result.applied is False
    assert result.detections[0].behavior is None
    assert len(result.detections) == 1


def test_enhancer_missing_template():
    settings = Settings(llm_enabled=True)
    enhancer = LlmBehaviorEnhancer(settings, client=_FakeClient([]), prompts={})
    result = enhancer.enhance(
        EnhanceRequest(
            behavior="UNKNOWN_BEH",
            image_bgr=_blank_image(),
            detections=[_person(1, 1, 10, 10)],
            algorithm_code="UNKNOWN_BEH",
        )
    )
    assert result.applied is False
