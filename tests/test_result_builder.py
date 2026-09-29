from app.detectors.base import RawDetection
from app.result_builder import build_result, build_zero_result, filter_for_code
from app.schemas import AiInferenceRequest


def _request(types: list[str]) -> AiInferenceRequest:
    return AiInferenceRequest.model_validate(
        {
            "requestId": "r1",
            "taskId": "t1",
            "deviceId": "d1",
            "algorithmTypes": types,
            "bucket": "detection-frames",
            "imageObjectKey": "a.jpg",
            "timestamp": 1000,
        }
    )


def test_filter_for_engineering():
    dets = [
        RawDetection("excavator", "excavator", 0.9, [1, 2, 3, 4], "excavator"),
        RawDetection("person", "person", 0.8, [5, 6, 7, 8], "person"),
    ]
    matched = filter_for_code("ENGINEERING_VEHICLE_DETECTION", dets)
    assert len(matched) == 1
    assert matched[0].object_type == "excavator"


def test_build_result_slots_aligned():
    dets = [
        RawDetection("person", "person", 0.9, [10, 20, 30, 40], "person"),
        RawDetection("excavator", "excavator", 0.85, [1, 2, 3, 4], "excavator"),
    ]
    result = build_result(
        _request(["FISHING_DETECTION", "ENGINEERING_VEHICLE_DETECTION"]),
        dets,
    )
    assert result.algorithm_type == [
        "FISHING_DETECTION",
        "ENGINEERING_VEHICLE_DETECTION",
    ]
    assert result.algorithm_num == [1, 1]
    assert result.image_url == ["", ""]
    assert len(result.detections) == 2
    payload = result.to_kafka_dict()
    assert payload["algorithmType"] == result.algorithm_type
    assert payload["algorithmNum"] == [1, 1]
    assert "detections" in payload
    assert payload["detections"][0]["bbox"] == [10, 20, 30, 40]


def test_build_result_unknown_code_zero():
    result = build_result(_request(["NO_SUCH"]), [])
    assert result.algorithm_type == ["NO_SUCH"]
    assert result.algorithm_num == [0]


def test_build_zero_result():
    result = build_zero_result(_request(["VEHICLE_DETECTION"]), reason="fail")
    assert result.algorithm_num == [0]
    assert result.detections == []
