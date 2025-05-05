from ultralytics import YOLO
import numpy as np
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

class YOLOv8Ultralytics:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, conf_threshold=0.5):
        """
        image: np.ndarray (BGR, as from OpenCV)
        Returns: list of [x1, y1, x2, y2, conf, class_id]
        """
        # Save original size
        orig_h, orig_w = image.shape[:2]
        # Ultralytics expects RGB
        img_rgb = image[..., ::-1]
        results = self.model(img_rgb, conf=conf_threshold)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box in YOLO's inference size
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Get model input size (assume square)
                inf_w, inf_h = r.orig_shape[1], r.orig_shape[0]
                # Rescale to original webcam size
                x1 = int(x1 * orig_w / inf_w)
                x2 = int(x2 * orig_w / inf_w)
                y1 = int(y1 * orig_h / inf_h)
                y2 = int(y2 * orig_h / inf_h)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, class_id])
        return detections