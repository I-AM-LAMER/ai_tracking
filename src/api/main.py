import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
from pydantic import BaseModel
import time

from src.tracking.yolov5 import YOLOv5
from src.tracking.utils import check_img_size, draw_detections, colors

# Constants for file handling
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

current_dir = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(current_dir.parent / 'logs' / 'app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/process-video":
            request.scope["max_request_size"] = MAX_FILE_SIZE
        return await call_next(request)

app.add_middleware(LargeFileMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

tracker = DeepSort(max_age=60, n_init=5)
model = YOLOv5(current_dir.parent / "models" / "crowdhuman.onnx", conf_thres=0.45, iou_thres=0.45, max_det=1000)
img_size = check_img_size([640, 640], s=model.stride)
track_history = {}

class LogEntry(BaseModel):
    level: str
    message: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Executed on application startup"""
    logger.info("Starting up the application")
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Model path: {current_dir.parent / 'models' / 'crowdhuman.onnx'}")

last_time = [time.time()]
fps = [0.0]
frame_counter = [0]
YOLO_INTERVAL = 5
last_detections = [[]] 

def process_frame(frame: np.ndarray) -> np.ndarray:
    try:
        # FPS расчет
        now = time.time()
        fps[0] = 1.0 / (now - last_time[0]) if last_time[0] != 0 else 0.0
        last_time[0] = now

        resized_image = cv2.resize(frame, tuple(img_size))

        frame_counter[0] += 1

        if frame_counter[0] % YOLO_INTERVAL == 1 or not last_detections[0]:
            boxes, scores, class_ids = model(resized_image)
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                label = model.names[int(class_id)] if isinstance(model.names, dict) else str(class_id)
                detections.append(([x, y, w, h], score, label))
            last_detections[0] = detections
        else:
            detections = last_detections[0]

        tracks = tracker.update_tracks(detections, frame=resized_image)
        unique_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            track_conf = track.det_conf if track.det_conf is not None else 0.0
            unique_ids.add(track_id)

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((x_center, y_center))
            # Ограничим длину истории (например, 50 точек)
            if len(track_history[track_id]) > 50:
                track_history[track_id] = track_history[track_id][-50:]

            pts = track_history[track_id]
            for j in range(1, len(pts)):
                cv2.line(resized_image, pts[j - 1], pts[j], (0, 255, 255), 2)

            draw_detections(
                resized_image,
                [x1, y1, x2, y2],
                track_conf,
                f"id {track_id}",
                colors(0),
                unique_count=len(unique_ids)
            )

        cv2.putText(
            resized_image,
            f"Count: {len(unique_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        # FPS в правом верхнем углу
        cv2.putText(
            resized_image,
            f"FPS: {fps[0]:.1f}",
            (resized_image.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )

        logger.debug(f"Processed frame with {len(unique_ids)} unique objects, FPS: {fps[0]:.1f}")
        return resized_image
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise

templates = Jinja2Templates(directory=current_dir / "templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    logger.info("Index page requested")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    temp_path = current_dir / "temp_video.mp4"
    with open(temp_path, "wb") as f:
        while chunk := await file.read(CHUNK_SIZE):
            f.write(chunk)
    return {"status": "ok"}

@app.get("/video-stream")
def video_stream():
    temp_path = current_dir / "temp_video.mp4"
    if not temp_path.exists():
        raise HTTPException(status_code=404, detail="No video uploaded")
    def gen():
        cap = cv2.VideoCapture(str(temp_path))
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            result = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', result)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/log")
async def client_log(log_entry: LogEntry):
    """Endpoint for client-side logging"""
    log_level = getattr(logging, log_entry.level.upper(), logging.INFO)
    logger.log(log_level, f"Client: {log_entry.message}")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=9090, 
        log_level="info",
        limit_concurrency=100,
        limit_max_requests=100,
        timeout_keep_alive=600,
        loop="uvloop",
        http="httptools",
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_graceful_shutdown=300,
        server_header=False,
        date_header=False,
        access_log=True,
    )