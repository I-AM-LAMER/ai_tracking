import cv2
from src.tracking.detector import YOLOv8ONNX
from src.tracking.tracker import DeepSORTTracker
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "crowdhuman.onnx"
)

def process_video(input_path, output_path):
    import cv2
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = YOLOv8ONNX(MODEL_PATH, input_size=640)
    tracker = DeepSORTTracker()

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame=frame)
        for tr in tracks:
            x1, y1, x2, y2 = tr['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID: {tr['track_id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264")


if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection.mp4")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.mp4")
    process_video(input_path, output_path)
