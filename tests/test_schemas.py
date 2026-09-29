import json

from app.schemas import AiInferenceRequest, AiInferenceResult


def test_parse_request_from_java_json():
    raw = {
        "requestId": "abc",
        "taskId": "task-1",
        "deviceId": "dev-1",
        "algorithmTypes": ["ENGINEERING_VEHICLE_DETECTION"],
        "bucket": "detection-frames",
        "imageObjectKey": "frames/x.jpg",
        "timestamp": 1710000000000,
        "streamTimestampUs": 1,
        "capturedAtMs": 1710000000000,
    }
    req = AiInferenceRequest.model_validate(raw)
    assert req.request_id == "abc"
    assert req.image_object_key == "frames/x.jpg"
    assert req.required_ok()


def test_request_rejects_incomplete():
    req = AiInferenceRequest.model_validate(
        {
            "requestId": "abc",
            "taskId": "t",
            "deviceId": "d",
            "algorithmTypes": [],
        }
    )
    assert not req.required_ok()


def test_result_roundtrip_kafka_dict():
    result = AiInferenceResult.model_validate(
        {
            "requestId": "r",
            "taskId": "t",
            "deviceId": "d",
            "algorithmType": ["FISHING_DETECTION"],
            "algorithmNum": [0],
            "imageUrl": [""],
            "timestamp": 1,
            "detections": [],
        }
    )
    payload = result.to_kafka_dict()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["requestId"] == "r"
    assert loaded["algorithmType"] == ["FISHING_DETECTION"]
    assert loaded["algorithmNum"] == [0]
