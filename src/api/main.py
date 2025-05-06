import cv2
from src.tracking.detector import YOLOv8Ultralytics
from src.tracking.tracker import DeepSORTTracker
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "best.pt"
)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    detector = YOLOv8Ultralytics(MODEL_PATH)
    tracker = DeepSORTTracker()

    frame_id = 0
    last_tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame_resized = cv2.resize(frame, (target_w, target_h))
        # Обрабатываем только каждый 3-й кадр
        if frame_id % 3 == 0:
            detections = detector.detect(frame_resized)
            tracks = tracker.update(detections, frame=frame_resized)
            last_tracks = tracks
        else:
            tracks = last_tracks
        for tr in tracks:
            x1, y1, x2, y2 = tr['bbox']
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame_resized, f"ID: {tr['track_id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        out.write(frame_resized)
    cap.release()
    out.release()

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection.mp4")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.mp4")
    process_video(input_path, output_path)