import os
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import asyncio
import logging
from typing import List, Generator
from deep_sort_realtime.deepsort_tracker import DeepSort
from pydantic import BaseModel

from src.tracking.yolov5 import YOLOv5
from src.tracking.utils import check_img_size, scale_boxes, draw_detections, colors

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

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Настройки для загрузки больших файлов
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Увеличиваем лимит до 1GB
    request.scope["max_request_size"] = 1024 * 1024 * 1024  # 1GB
    response = await call_next(request)
    return response

tracker = DeepSort(max_age=60, n_init=5)
model = YOLOv5(current_dir.parent / "models" / "crowdhuman.onnx", conf_thres=0.45, iou_thres=0.45, max_det=1000)
img_size = check_img_size([640, 640], s=model.stride)

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

async def process_frame(frame: np.ndarray) -> np.ndarray:
    """Обработка одного кадра"""
    try:
        resized_image = cv2.resize(frame, tuple(img_size))
        logger.debug(f"Frame resized to {tuple(img_size)}")
        
        boxes, scores, class_ids = model(resized_image)
        boxes = scale_boxes(resized_image.shape, boxes, frame.shape).round()

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            label = model.names[int(class_id)]
            detections.append(([x, y, w, h], score, label))

        tracks = tracker.update_tracks(detections, frame=frame)
        unique_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_conf = track.det_conf if track.det_conf is not None else 0.0
            unique_ids.add(track_id)

            draw_detections(
                frame,
                [x1, y1, x2, y2],
                track_conf,
                f"id {track_id}",
                colors(0),
                unique_count=len(unique_ids)
            )

        cv2.putText(
            frame,
            f"Count: {len(unique_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )
        
        logger.debug(f"Processed frame with {len(unique_ids)} unique objects")
        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise

def frame_generator(video_bytes: bytes) -> Generator[bytes, None, None]:
    """Генератор кадров из видео"""
    logger.info("Starting video processing")
    temp_file = "temp_video.mp4"
    
    try:
        with open(temp_file, "wb") as f:
            f.write(video_bytes)
        logger.debug("Temporary video file created")
        
        cap = cv2.VideoCapture(temp_file)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            processed_frame = asyncio.run(process_frame(frame))
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        logger.error(f"Error in frame generator: {str(e)}")
        raise
    finally:
        cap.release()
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.debug("Temporary video file removed")

templates = Jinja2Templates(directory=current_dir / "templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    logger.info("Index page requested")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """Эндпоинт для обработки видео"""
    logger.info(f"Received video upload request: {file.filename}")
    try:
        contents = await file.read()
        logger.info(f"Video file read successfully, size: {len(contents)} bytes")
        return StreamingResponse(
            frame_generator(contents),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

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
        # Увеличиваем размер буфера и таймауты
        loop="uvloop",
        http="httptools",
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )