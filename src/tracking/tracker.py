from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class DeepSORTTracker:
    def __init__(self):
        # Initialize DeepSort tracker (use default parameters)
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
        self.last_tracks = []

    def update(self, detections, frame=None):
        """
        detections: list of [x1, y1, x2, y2, conf, class_id]
        Only tracks people (class_id == 0)
        Returns: list of dicts with keys: 'track_id', 'bbox', 'confidence'
        """
        # Filter for people
        people = [det for det in detections if det[5] == 0]
        if not people:
            return []

        # Prepare detections for DeepSort: [[x1, y1, x2, y2, confidence], ...]
        dets = np.array([det[:5] for det in people])
        # DeepSort expects: list of [ [x1, y1, x2, y2], confidence, class ]
        formatted = [
            (det[:4], det[4], "person") for det in dets
        ]
        tracks = self.tracker.update_tracks(formatted, frame=frame)
        self.last_tracks = [
            {
                "track_id": int(track.track_id),
                "bbox": [int(x) for x in track.to_ltrb()],
                "confidence": float(track.det_conf) if track.det_conf is not None else 0.0
            }
            for track in tracks if track.is_confirmed()
        ]
        return self.last_tracks

    def get_tracks(self):
        return self.last_tracks