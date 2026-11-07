# Real-Time AI Object Detection

Real-time object detection from your webcam, served through a web dashboard
with live statistics. Powered by **YOLOv8**, **OpenCV**, and **Flask**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-00B4D8)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Live webcam detection** — 80 COCO classes (person, car, dog, laptop, …) out of the box.
- **Browser dashboard** — MJPEG video stream with real-time stats, served by Flask.
- **Statistics** — live FPS sparkline, on-screen counts, running totals, top classes.
- **Temporal stabilizer** — IoU-based tracker that suppresses one-frame false positives
  (e.g. paper flashed as "cell phone", finger as "toothbrush"). Confirmed detections
  stay stable across frames with EMA-smoothed confidence.
- **Class filtering** — restrict detection to a whitelist by name
  (`--classes person,car,dog`) so you don't get irrelevant classes.
- **Thread-safe pipeline** — capture + inference on a worker thread, the web layer stays snappy.
- **Pluggable model** — default is `yolov8s` for accuracy; swap for `yolov8n/m/l/x` or any custom `.pt`.
- **Configurable via CLI or environment** — model, camera, confidence, image size, device.
- **CLI demo & benchmark** — headless scripts and a static-image demo for quick tests.

---

## Quick start

### Windows — one click

Double-click **`start.bat`**. First run creates a virtual env, installs
dependencies and launches the dashboard; next runs just launch it. A browser
tab opens automatically on <http://127.0.0.1:5000>.

For the headless OpenCV-window demo instead, double-click **`start-cli.bat`**.

### Manual (any OS)

```bash
git clone https://github.com/<your-username>/real-time-ai-object-detection.git
cd real-time-ai-object-detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

python run.py
```

Then open <http://127.0.0.1:5000>. The first run downloads the YOLOv8-nano
weights (~6 MB) automatically.

### Common options

```bash
python run.py --model yolov8n.pt                 # smaller / faster model
python run.py --source 1                         # second webcam
python run.py --source path/to/video.mp4         # video file instead of a camera
python run.py --classes person,car,dog           # only show these classes
python run.py --conf 0.6                         # stricter confidence filter
python run.py --augment                          # test-time augmentation (slower, more accurate)
python run.py --device cuda                      # use an NVIDIA GPU if available
python run.py --no-stabilize                     # disable the temporal stabilizer
```

---

## Dashboard

The web UI shows:

| Panel | What it shows |
|---|---|
| **Video** | Live annotated feed with colored boxes and per-object confidence. |
| **FPS card** | Current FPS and a rolling sparkline of recent frames. |
| **Objects on screen** | How many objects are visible right now, with a breakdown. |
| **Frames processed** | Lifetime frame count and cumulative detection total. |
| **Unique classes seen** | How many distinct COCO classes have shown up. |
| **Top detected classes** | Horizontal bars of the most-seen classes since start. |
| **On screen right now** | Chips for every class currently visible, with counts. |

A **Reset statistics** button clears the running totals without restarting the stream.

---

## Project layout

```
.
├── run.py                   # dashboard entry point
├── requirements.txt
├── src/
│   ├── detector.py          # YOLOv8 wrapper + annotation
│   ├── camera.py            # OpenCV capture helper
│   ├── stabilizer.py        # IoU-based temporal filter (anti-flicker)
│   ├── stats.py             # thread-safe stats tracker
│   ├── stream.py            # worker thread: capture → infer → stabilize → publish
│   ├── types.py             # shared dataclasses (no cv2/torch deps)
│   └── app.py               # Flask app + JSON APIs
├── templates/index.html     # dashboard page
├── static/
│   ├── style.css            # dashboard styling
│   └── dashboard.js         # polls /api/stats and updates the UI
├── examples/
│   ├── detect_cli.py        # headless OpenCV window demo
│   ├── demo_image.py        # static-image demo (no webcam needed)
│   └── benchmark.py         # throughput benchmark
├── tests/
│   ├── test_stats.py        # unit tests for the stats tracker
│   └── test_stabilizer.py   # unit tests for the temporal filter
└── docs/                    # extra notes & screenshots
```

---

## HTTP API

The dashboard is driven by a tiny JSON API you can also use from your own code:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard page. |
| `GET` | `/video_feed` | MJPEG stream of annotated frames. |
| `GET` | `/api/stats` | Live stats snapshot (FPS, counts, totals, uptime). |
| `GET` | `/api/detections` | Bounding boxes for the latest processed frame. |
| `POST` | `/api/reset` | Reset running statistics. |
| `GET` | `/api/health` | Liveness probe. |

Example:

```bash
curl http://127.0.0.1:5000/api/stats
```

```json
{
  "fps": 28.4,
  "frames_processed": 842,
  "uptime_seconds": 31.2,
  "current_counts": { "person": 1, "laptop": 1 },
  "total_counts":   { "person": 820, "laptop": 410, "cup": 55 },
  "top_classes":    [["person", 820], ["laptop", 410], ["cup", 55]],
  "recent_fps":     [27.9, 28.1, 28.3, 28.4]
}
```

---

## Configuration

Any option can be set via `--flag` or environment variable:

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--model` | `YOLO_MODEL` | `yolov8s.pt` | Any Ultralytics-compatible weights file. |
| `--conf` | `YOLO_CONF` | `0.5` | Minimum confidence to keep a detection. |
| `--imgsz` | `YOLO_IMGSZ` | `640` | Inference resolution. Higher = better small-object accuracy, slower. |
| `--augment` | `YOLO_AUGMENT` | off | Test-time augmentation (TTA). Slower but more accurate. |
| `--classes` | `YOLO_CLASSES` | *(all)* | Comma-separated class names, e.g. `person,car,dog`. |
| `--no-stabilize` | `STABILIZE=0` | stabilize on | Disable the IoU-based temporal filter. |
| — | `STABILIZE_MIN_HITS` | `3` | Frames a detection must appear in before being shown. |
| — | `STABILIZE_MAX_MISSES` | `5` | Frames a confirmed detection survives after disappearing. |
| — | `YOLO_IOU` | `0.45` | NMS IoU threshold. |
| `--device` | `YOLO_DEVICE` | *(auto)* | `cpu`, `cuda`, `cuda:0`, `mps`, … |
| `--source` | `CAMERA_SOURCE` | `0` | Camera index or a video file path. |
| — | `CAMERA_WIDTH` | `1280` | Capture width in pixels. |
| — | `CAMERA_HEIGHT` | `720` | Capture height in pixels. |
| — | `CAMERA_FPS` | `30` | Requested capture FPS. |

---

## CLI demo (no browser)

```bash
python examples/detect_cli.py             # webcam
python examples/detect_cli.py --source path/to/video.mp4 --model yolov8s.pt
```

Press **Q** to quit the OpenCV window.

## Static-image demo (no webcam)

Downloads a sample image, runs detection, and writes `demo_output.jpg`.
Handy for verifying accuracy without the camera:

```bash
python examples/demo_image.py
python examples/demo_image.py --image path/to/your.jpg --classes person,car
```

## Benchmark

```bash
python examples/benchmark.py --model yolov8n.pt --iters 200
```

Prints achievable FPS for your CPU/GPU on a synthetic 1280×720 input.

---

## Tests

```bash
python tests/test_stats.py
# or, with pytest installed:
pytest tests/
```

---

## Model options

| Model | Params | Speed (CPU) | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 3.2M | fastest | lowest |
| `yolov8s.pt` | 11.2M | fast | **default** — good balance |
| `yolov8m.pt` | 25.9M | medium | high |
| `yolov8l.pt` | 43.7M | slow | higher |
| `yolov8x.pt` | 68.2M | slowest | highest |

The `s` model is the default — it's still CPU-friendly but notably more accurate
than `n`. On a weak laptop drop to `yolov8n.pt`; with an NVIDIA GPU jump to
`yolov8m.pt` for great accuracy and still real-time speeds.

---

## Troubleshooting

- **"Could not open video source 0"** — the webcam is busy (another app has it),
  or on Windows the DirectShow backend failed. Try `--source 1` or close other
  camera-using apps.
- **Very low FPS** — switch to `--model yolov8n.pt`, drop `--imgsz 480`,
  lower capture resolution (`CAMERA_WIDTH=640 CAMERA_HEIGHT=480`), or use a
  GPU with `--device cuda`.
- **Objects misclassified (paper → "cell phone", finger → "toothbrush")** —
  COCO only knows 80 classes; anything outside is forced into the nearest look-alike.
  Raise `--conf` to 0.6+, use `--classes` to whitelist only what matters,
  or train a custom YOLOv8 for your own classes.
- **Flickery / "blinking" detections** — this is what the stabilizer is for;
  it's enabled by default. If you want every raw frame's output, pass
  `--no-stabilize`. To tune, set `STABILIZE_MIN_HITS` / `STABILIZE_MAX_MISSES`.
- **Firewall prompt on first run** — Flask opens a local port; allow it on the
  loopback interface only.

---

## Tech stack

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — detection model
- [OpenCV](https://opencv.org/) — webcam capture & drawing
- [Flask](https://flask.palletsprojects.com/) — web server & JSON API
- Vanilla HTML/CSS/JS — zero-build dashboard

---

## License

[MIT](LICENSE)
