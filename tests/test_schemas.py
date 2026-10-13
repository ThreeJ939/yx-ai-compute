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
            "timestamp": 1,
            "results": [
                {
                    "algorithmType": "BEH_FISHING",
                    "algorithmNum": 0,
                    "imageUrl": "",
                    "detections": [],
                }
            ],
        }
    )
    payload = result.to_kafka_dict()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["requestId"] == "r"
    assert loaded["results"][0]["algorithmType"] == "BEH_FISHING"
    assert loaded["results"][0]["algorithmNum"] == 0
    assert "stage" not in loaded
    assert "partial" not in loaded
    assert "algorithmType" not in loaded
