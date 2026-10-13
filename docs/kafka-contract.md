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
  "algorithmTypes": ["ENGINEERING_VEHICLE_DETECTION", "BEH_FISHING"],
  "bucket": "detection-frames",
  "imageObjectKey": "frames/task-1/xxx.jpg",
  "timestamp": 1710000000000,
  "streamTimestampUs": 123456,
  "capturedAtMs": 1710000000000
}
```

Image bytes are **not** in the message. Compute downloads via MinIO `bucket` + `imageObjectKey`.

## Result (`AiInferenceResult`) — nested per algorithm

Recognition `AiResultConsumer` requires `results` non-empty:

```json
{
  "requestId": "abc123",
  "taskId": "task-1",
  "deviceId": "cam-01",
  "timestamp": 1710000000500,
  "results": [
    {
      "algorithmType": "ENGINEERING_VEHICLE_DETECTION",
      "algorithmNum": 2,
      "imageUrl": "https://minio/.../abc123.jpg",
      "detections": [
        {
          "detId": "a1b2c3d4e5f6",
          "objectType": "excavator",
          "label": "excavator",
          "confidence": 0.91,
          "bbox": [100.0, 80.0, 200.0, 150.0]
        }
      ]
    },
    {
      "algorithmType": "BEH_FISHING",
      "algorithmNum": 1,
      "imageUrl": "",
      "detections": [
        {
          "detId": "c3d4e5f6a1b2",
          "objectType": "person",
          "label": "person",
          "confidence": 0.82,
          "bbox": [50.0, 60.0, 80.0, 200.0],
          "behavior": "fishing"
        }
      ]
    }
  ]
}
```

- `bbox`: `[x, y, width, height]` in pixels
- `imageUrl`: original inference frame (`http(s)://{MINIO_ENDPOINT}/{bucket}/{imageObjectKey}`); same for all slots. Not an annotated image — recognition draws boxes and uploads alarm snapshots itself
- `DET_*` / pure detect codes: always present in `results` with YOLO boxes
- `BEH_*` + LLM disabled: slot omitted entirely (no behavior alarm)
- `BEH_*` + LLM enabled: only detections with non-empty `behavior`; empty confirmed set still emits `algorithmNum=0`
- YOLO intermediate / preview results go via WebSocket, not this Kafka topic
- No `stage` / `partial` fields

## Failure behavior

| Case | Compute behavior |
|------|------------------|
| Poison / invalid JSON | Log, commit offset (skip) |
| Missing required fields | Log, commit |
| MinIO transient error | Log, **do not** commit (retry) |
| Decode / infer error | Publish zero `algorithmNum` slots (skip BEH_* if LLM off), commit |

## Algorithm codes

See `GET /capabilities` or `app/registry.py`.
