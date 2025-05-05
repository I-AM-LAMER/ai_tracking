from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from src.tracking.detector import YOLOv8Ultralytics
from src.tracking.tracker import DeepSORTTracker
from src.visualization.server import router as visualization_router
import logging
import os
from contextlib import asynccontextmanager
import os
from fastapi.responses import StreamingResponse
import cv2


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield

app = FastAPI(lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'src', 'models', 'yolov8n.pt')
MODEL_PATH = os.path.abspath(MODEL_PATH)

detector = YOLOv8Ultralytics(MODEL_PATH)

tracker = DeepSORTTracker()

class TrackResponse(BaseModel):
    tracks: list

from fastapi.responses import StreamingResponse

def process_video_stream(input_file):
    import cv2
    import tempfile

    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Уменьшаем разрешение для снижения нагрузки (например, 640x360)
    target_w, target_h = 640, 360

    with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_out:
        out = cv2.VideoWriter(temp_out.name, fourcc, fps, (target_w, target_h))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            # Пропускаем кадры для ускорения (обрабатываем каждый 3-й)
            if frame_id % 3 != 0:
                resized = cv2.resize(frame, (target_w, target_h))
                out.write(resized)
                continue
            resized = cv2.resize(frame, (target_w, target_h))
            detections = detector.detect(resized)
            tracks = tracker.update(detections, frame=resized)
            for tr in tracks:
                x1, y1, x2, y2 = tr['bbox']
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(resized, f"ID: {tr['track_id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            out.write(resized)
        cap.release()
        out.release()
        temp_out.seek(0)
        while True:
            chunk = temp_out.read(1024 * 1024)
            if not chunk:
                break
            yield chunk

@app.post("/process_video")
async def process_video(video: UploadFile = File(...)):
    import tempfile
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as in_tmp:
        shutil.copyfileobj(video.file, in_tmp)
        in_path = in_tmp.name

    return StreamingResponse(process_video_stream(in_path), media_type='video/mp4')

app.include_router(visualization_router)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualization", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")