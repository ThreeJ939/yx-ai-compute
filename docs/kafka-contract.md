# Kafka contract between yx-ai-recognition and yx-ai-compute

## Topics

| Topic | Direction | Key |
|-------|-----------|-----|
| `ai.inference.request` | recognition → compute | `taskId` |
| `ai.inference.result` | compute → recognition | `taskId` |

Consumer group (compute): `yx-ai-compute-group`  
Consumer group (recognition result): configured in `ai-recognition.kafka.consumer-group`

## Request (`AiInferenceRequest`)

```json
{
  "requestId": "abc123",
  "taskId": "task-1",
  "deviceId": "cam-01",
  "algorithmTypes": ["ENGINEERING_VEHICLE_DETECTION", "FISHING_DETECTION"],
  "bucket": "detection-frames",
  "imageObjectKey": "frames/task-1/xxx.jpg",
  "timestamp": 1710000000000,
  "streamTimestampUs": 123456,
  "capturedAtMs": 1710000000000
}
```

Image bytes are **not** in the message. Compute downloads via MinIO `bucket` + `imageObjectKey`.

## Result (`AiInferenceResult`) — Java-compatible slots

Recognition `AiResultConsumer` requires aligned lists with size > 0:

```json
{
  "requestId": "abc123",
  "taskId": "task-1",
  "deviceId": "cam-01",
  "algorithmType": ["ENGINEERING_VEHICLE_DETECTION", "FISHING_DETECTION"],
  "algorithmNum": [2, 1],
  "imageUrl": ["", ""],
  "timestamp": 1710000000500,
  "detections": [
    {
      "detId": "a1b2c3d4e5f6",
      "objectType": "excavator",
      "label": "excavator",
      "confidence": 0.91,
      "bbox": [100.0, 80.0, 200.0, 150.0],
      "algorithmType": "ENGINEERING_VEHICLE_DETECTION"
    }
  ],
  "stage": "yolo",
  "partial": false
}
```

- `bbox`: `[x, y, width, height]` in pixels  
- `detections` / `stage` / `partial`: forward-compatible; current Java DTO ignores unknown fields  
- Phase-2 may emit two results per `requestId` (`partial=true` then `false`); Java must change idempotency first  

## Failure behavior

| Case | Compute behavior |
|------|------------------|
| Poison / invalid JSON | Log, commit offset (skip) |
| Missing required fields | Log, commit |
| MinIO transient error | Log, **do not** commit (retry) |
| Decode / infer error | Publish zero `algorithmNum` slots, commit |

## Algorithm codes (phase-1)

See `GET /capabilities` or `app/registry.py`.
