from app.detectors.base import RawDetection
from app.result_builder import (
    build_result,
    build_source_image_url,
    build_zero_result,
    filter_for_code,
    has_positive_detections,
)
from app.schemas import AiInferenceRequest

_SOURCE_URL = "http://localhost:9000/detection-frames/a.jpg"


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


def _build(request, dets, **kwargs):
    return build_result(
        request,
        dets,
        minio_endpoint="localhost:9000",
        minio_secure=False,
        **kwargs,
    )


def test_filter_for_engineering():
    dets = [
        RawDetection("excavator", "excavator", 0.9, [1, 2, 3, 4], "excavator"),
        RawDetection("person", "person", 0.8, [5, 6, 7, 8], "person"),
    ]
    matched = filter_for_code("ENGINEERING_VEHICLE_DETECTION", dets)
    assert len(matched) == 1
    assert matched[0].object_type == "excavator"


def test_build_source_image_url_http():
    assert (
        build_source_image_url(
            _request(["DET_PERSON"]), endpoint="localhost:9000", secure=False
        )
        == _SOURCE_URL
    )


def test_build_source_image_url_s3_fallback():
    assert build_source_image_url(_request(["DET_PERSON"])) == (
        "s3://detection-frames/a.jpg"
    )


def test_build_result_nested_slots():
    dets = [
        RawDetection("person", "person", 0.9, [10, 20, 30, 40], "person"),
        RawDetection("excavator", "excavator", 0.85, [1, 2, 3, 4], "excavator"),
    ]
    # DET_PERSON is detect-stage; person boxes appear without needing LLM
    result = _build(
        _request(["DET_PERSON", "ENGINEERING_VEHICLE_DETECTION"]),
        dets,
        llm_enabled=False,
    )
    assert len(result.results) == 2
    assert result.results[0].algorithm_type == "DET_PERSON"
    assert result.results[0].algorithm_num == 1
    assert result.results[1].algorithm_type == "ENGINEERING_VEHICLE_DETECTION"
    assert result.results[1].algorithm_num == 1
    assert result.results[0].image_url == _SOURCE_URL
    assert result.results[1].image_url == _SOURCE_URL
    assert len(result.results[0].detections) == 1
    assert len(result.results[1].detections) == 1

    payload = result.to_kafka_dict()
    assert "results" in payload
    assert payload["results"][0]["algorithmType"] == "DET_PERSON"
    assert payload["results"][0]["algorithmNum"] == 1
    assert payload["results"][0]["imageUrl"] == _SOURCE_URL
    assert payload["results"][0]["detections"][0]["bbox"] == [10, 20, 30, 40]
    assert "stage" not in payload
    assert "partial" not in payload
    assert "algorithmType" not in payload


def test_build_result_beh_with_llm_only_confirmed():
    dets = [
        RawDetection(
            "person", "person", 0.9, [10, 20, 30, 40], "person", behavior="fishing"
        ),
        RawDetection("person", "person", 0.8, [50, 60, 70, 80], "person"),
    ]
    result = _build(_request(["BEH_FISHING"]), dets, llm_enabled=True)
    assert len(result.results) == 1
    assert result.results[0].algorithm_type == "BEH_FISHING"
    assert result.results[0].algorithm_num == 1
    assert result.results[0].detections[0].behavior == "fishing"
    assert result.results[0].image_url == _SOURCE_URL


def test_build_result_beh_llm_disabled_skips_slot():
    dets = [
        RawDetection("person", "person", 0.9, [10, 20, 30, 40], "person"),
        RawDetection("excavator", "excavator", 0.85, [1, 2, 3, 4], "excavator"),
    ]
    result = _build(
        _request(["BEH_FISHING", "ENGINEERING_VEHICLE_DETECTION"]),
        dets,
        llm_enabled=False,
    )
    assert len(result.results) == 1
    assert result.results[0].algorithm_type == "ENGINEERING_VEHICLE_DETECTION"
    assert result.results[0].algorithm_num == 1
    assert result.results[0].image_url == _SOURCE_URL


def test_build_result_beh_llm_enabled_all_negative_zero_slot():
    dets = [
        RawDetection("person", "person", 0.9, [10, 20, 30, 40], "person"),
    ]
    result = _build(_request(["BEH_FISHING"]), dets, llm_enabled=True)
    assert len(result.results) == 1
    assert result.results[0].algorithm_type == "BEH_FISHING"
    assert result.results[0].algorithm_num == 0
    assert result.results[0].detections == []
    assert result.results[0].image_url == _SOURCE_URL
    assert has_positive_detections(result) is False


def test_build_result_unknown_code_zero():
    result = _build(_request(["NO_SUCH"]), [])
    assert len(result.results) == 1
    assert result.results[0].algorithm_type == "NO_SUCH"
    assert result.results[0].algorithm_num == 0
    assert result.results[0].image_url == _SOURCE_URL
    assert has_positive_detections(result) is False


def test_has_positive_detections_true():
    dets = [
        RawDetection("person", "person", 0.9, [10, 20, 30, 40], "person"),
    ]
    result = _build(_request(["DET_PERSON"]), dets)
    assert has_positive_detections(result) is True


def test_build_zero_result():
    result = build_zero_result(
        _request(["VEHICLE_DETECTION"]),
        reason="fail",
        minio_endpoint="localhost:9000",
    )
    assert len(result.results) == 1
    assert result.results[0].algorithm_num == 0
    assert result.results[0].detections == []
    assert result.results[0].image_url == _SOURCE_URL
    assert has_positive_detections(result) is False


def test_build_zero_result_skips_beh_when_llm_off():
    result = build_zero_result(
        _request(["BEH_FISHING", "VEHICLE_DETECTION"]),
        reason="fail",
        llm_enabled=False,
        minio_endpoint="localhost:9000",
    )
    assert len(result.results) == 1
    assert result.results[0].algorithm_type == "VEHICLE_DETECTION"
    assert result.results[0].image_url == _SOURCE_URL
