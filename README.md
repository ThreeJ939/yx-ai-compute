# yx-ai-compute

独立 Python 算力微服务：消费 `yx-ai-recognition` 发出的 Kafka 推理请求，从 MinIO 拉帧，使用 YOLO-World 按 `algorithmTypes` 检测，回写兼容现有 `AiInferenceResult` 槽位的结果。

## 架构位置

```
yx-ai-recognition  --ai.inference.request-->  yx-ai-compute
       ^                                          |
       |         ai.inference.result              | MinIO get_object
       +------------------------------------------+
```

- 服务名：`ai-compute-service`
- Consumer Group：`yx-ai-compute-group`
- Health：`http://0.0.0.0:18100/health`
- Capabilities：`http://0.0.0.0:18100/capabilities`
- Preview：`http://0.0.0.0:18100/preview`（图片 + 视频流：本地 MP4 / 浏览器摄像头 / RTSP）

## 快速开始

```bash
cd yx-ai-compute
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：Kafka / MinIO / YOLO_WORLD_WEIGHTS / DEVICE

# 下载 YOLO-World 权重到 weights/（示例）
# 首次运行 ultralytics 也会按 YOLO_WORLD_WEIGHTS 路径尝试加载/下载

python -m app.main
```

## 环境变量

见 [`.env.example`](.env.example)。关键项：

| 变量 | 说明 |
|------|------|
| `KAFKA_ENABLED` | 是否启动 Kafka 消费（`false` = 仅 HTTP/预览，本地联调） |
| `KAFKA_BOOTSTRAP_SERVERS` | 与 recognition 相同 |
| `KAFKA_USERNAME` / `KAFKA_PASSWORD` | SASL（无则留空并改 `KAFKA_SECURITY_PROTOCOL=PLAINTEXT`） |
| `MINIO_ENDPOINT` | 直连 MinIO，格式 `host:port` |
| `YOLO_WORLD_WEIGHTS` | 检测权重路径，勿提交到 git |
| `YOLO_CLIP_WEIGHTS` | CLIP `ViT-B-32.pt` 本地路径（离线 `set_classes`）；空则自动下载 |
| `DEVICE` | `cuda` 或 `cpu` |
| `PREVIEW_ENABLED` | 是否启用 `/preview` 与 `/preview/stream` |
| `PREVIEW_INTERVAL_MS` | 视频流默认抽帧间隔（CPU 建议 800–2000） |
| `CONF_THRESHOLD` | 检测置信度阈值 |
| `YOLO_TILED` | 是否滑窗推理（默认 true，利于远景/小目标） |
| `YOLO_TILE_SIZE` / `YOLO_TILE_OVERLAP` | 切块边长与重叠率（对齐 yolo-world-demo） |
| `YOLO_IMGSZ` | 单块/整图 `predict` 的 imgsz |

图宽或高大于 `YOLO_TILE_SIZE` 时走滑窗 + 按类 NMS；否则整图一次推理。

CLIP 可复用 demo 文件，例如：

```text
YOLO_CLIP_WEIGHTS=D:/java_workspace/YongXin/yolo-world-demo/weights/clip/ViT-B-32.pt
```

或复制到本仓 `weights/clip/ViT-B-32.pt`。服务启动后会链到 Ultralytics / `~/.cache/clip`，避免断网下载失败。

## 算法路由（一期）

| algorithmTypes code | YOLO-World prompts |
|---------------------|--------------------|
| `ENGINEERING_VEHICLE_DETECTION` | excavator, bulldozer, truck, crane, ... |
| `FISHING_DETECTION` | person（行为增强一期 noop） |
| `VEHICLE_DETECTION` | car, truck, bus |
| 未知 code | 槽位 `algorithmNum=0`，不中断其它算法 |

同一请求内多个检测类会合并 prompts **一次前向**，再按 code 过滤填槽。

## 结果契约

兼容 Java `AiInferenceResult`：

- `algorithmType` / `algorithmNum` / `imageUrl` 三列表对齐
- 额外字段 `detections`（bbox `[x,y,w,h]`）；当前 recognition 会忽略未知字段

详见 [docs/kafka-contract.md](docs/kafka-contract.md)。

## 测试

```bash
pip install pytest
pytest -q
```

### 本地图片直调（不依赖 Kafka / MinIO）

在项目根目录、已配置 `.env` 与权重后：

```bash
# 按算法字典 code（默认工程车+人员）
python -m scripts.infer_local path/to/image.jpg

# 指定算法
python -m scripts.infer_local image.jpg -a ENGINEERING_VEHICLE_DETECTION,DET_SHIP

# 直接写提示词，并保存画框图
python -m scripts.infer_local image.jpg -p "person,boat,excavator" -o out.jpg

# 覆盖置信度、关闭滑窗
python -m scripts.infer_local image.jpg --conf 0.15 --no-tile -v
```

脚本会打印与 Kafka 结果同结构的 JSON（含 `algorithmType` / `algorithmNum` / `detections`）。

### 本地 MP4 压测（实时性能）

```bash
# 默认约每 2s 视频时间抽一帧推理（按源 fps 换算 sample-every），最多 50 次
python -m scripts.bench_mp4 path/to/video.mp4

# 每 25 帧推理一次、关闭滑窗、排除首次加载
python -m scripts.bench_mp4 video.mp4 --sample-every 25 --no-tile --warmup 1 -v

# 模拟 2s 间隔、只跑 30 次、保存预览帧
python -m scripts.bench_mp4 video.mp4 --interval-ms 2000 --max-infer 30 \
  --preview-dir /tmp/bench_preview -a DET_PERSON,ENGINEERING_VEHICLE_DETECTION
```

输出含 avg/p50/p95 推理耗时、effective FPS，以及是否跟得上目标抽帧间隔（`keep_up`）。

## Docker

```bash
docker build -t yx-ai-compute:0.1.0 .
docker run --rm \
  --env-file .env \
  -v /path/to/weights:/app/weights \
  -p 18100:18100 \
  yx-ai-compute:0.1.0
```

## 联调验收

1. `.env` 与 recognition 使用同一 Kafka / MinIO
2. recognition 启用含 `ENGINEERING_VEHICLE_DETECTION` 的任务并出帧
3. 本服务日志：consume → download → infer → produce
4. recognition 侧出现 `RecognitionEvent`；有目标时 `algorithmNum>0`
5. `curl http://localhost:18100/health` 与 `/capabilities`

## 本期不做

- 改 Java 两阶段幂等 / 消费 detections
- 真实 LLM/VLM
- Nacos / 网关 Ability
- 人员专用闭集模型
