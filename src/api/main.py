import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

from typing import List

from src.tracking.yolov5 import YOLOv5
from src.tracking.utils import check_img_size, scale_boxes, draw_detections, colors, increment_path, LoadMedia


def run_object_detection(
    weights: str,
    source: str,
    img_size: List[int],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    save: bool,
    view: bool,
    project: str,
    name: str
):
    if save:
        save_dir = increment_path(Path(project) / name)
        save_dir.mkdir(parents=True, exist_ok=True)

    tracker = DeepSort(max_age=60, n_init=5)
    model = YOLOv5(weights, conf_thres, iou_thres, max_det)
    img_size = check_img_size(img_size, s=model.stride)
    dataset = LoadMedia(source, img_size=img_size)
    unique_ids = set()

    vid_writer = None
    if save and dataset.type in ["video", "webcam"]:
        cap = dataset.cap
        save_path = str(save_dir / os.path.basename(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for resized_image, original_image, status in dataset:
        boxes, scores, class_ids = model(resized_image)
        boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            label = model.names[int(class_id)]
            detections.append(([x, y, w, h], score, label))

        tracks = tracker.update_tracks(detections, frame=original_image)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_conf = track.det_conf if track.det_conf is not None else 0.0

            unique_ids.add(track_id)

            draw_detections(
                original_image,
                [x1, y1, x2, y2],
                track_conf,
                f"id {track_id}",
                colors(0),
                unique_count=len(unique_ids)
            )
        cv2.putText(
            original_image,
            f"Count: {len(unique_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        for c in np.unique(class_ids):
            n = (class_ids == c).sum()
            status += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "

        if view:
            cv2.imshow('Webcam Inference', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(status)

        if save:
            if dataset.type == "image":
                save_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
                cv2.imwrite(save_path, original_image)
            elif dataset.type in ["video", "webcam"]:
                vid_writer.write(original_image)

    if save and vid_writer is not None:
        vid_writer.release()

    if save:
        print(f"Results saved to {save_dir}")

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yolov5m.onnx", help="model path")
    parser.add_argument("--source", type=str, default="0", help="Path to video/image/webcam")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.45, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--save", action="store_true", help="Save detected images")
    parser.add_argument("--view", action="store_true", help="View inferenced images")
    parser.add_argument("--project", default="runs", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    args = parser.parse_args()
    args.img_size = args.img_size * 2 if len(args.img_size) == 1 else args.img_size
    return args


def download_weights(weights):
    pass


def main():
    params = parse_args()
    run_object_detection(
        weights=params.weights,
        source=params.source,
        img_size=params.img_size,
        conf_thres=params.conf_thres,
        iou_thres=params.iou_thres,
        max_det=params.max_det,
        save=params.save,
        view=params.view,
        project=params.project,
        name=params.name
    )


if __name__ == "__main__":
    main()
