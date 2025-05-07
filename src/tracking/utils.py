import os
import cv2
import time
import math
import numpy as np
from pathlib import Path


class Colors:
    def __init__(self):
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def check_img_size(imgsz, s=32):
    def make_divisible(x, divisor): return math.ceil(x / divisor) * divisor

    imgsz = list(imgsz)
    new_size = [make_divisible(x, s) for x in imgsz]

    return new_size


def increment_path(path,  sep="", mkdir=False):
    path = Path(path)
    if path.exists():
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True)

    return path


def letterbox(image, target_shape=(640, 640), color=(114, 114, 114)):
    height, width = image.shape[:2]

    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_size = (int(width * scale), int(height * scale))

    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    dw, dh = (target_shape[1] - new_size[0]) / 2, (target_shape[0] - new_size[1]) / 2
    top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
    left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, scale, (dw, dh)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    scale = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # scale  = old / new
    dw, dh = (img1_shape[1] - img0_shape[1] * scale) / 2, (img1_shape[0] - img0_shape[0] * scale) / 2  # wh padding

    boxes[..., [0, 2]] -= dw  # x padding
    boxes[..., [1, 3]] -= dh  # y padding
    boxes[..., :4] /= scale

    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def get_txt_color(color=(128, 128, 128), txt_color=(255, 255, 255)):
    dark_colors = {
        (235, 219, 11),
        (243, 243, 243),
        (183, 223, 0),
        (221, 111, 255),
        (0, 237, 204),
        (68, 243, 0),
        (255, 255, 0),
        (179, 255, 1),
        (11, 255, 162),
    }
    light_colors = {
        (255, 42, 4),
        (79, 68, 255),
        (255, 0, 189),
        (255, 180, 0),
        (186, 0, 221),
        (0, 192, 38),
        (255, 36, 125),
        (104, 0, 123),
        (108, 27, 255),
        (47, 109, 252),
        (104, 31, 17),
    }

    if color in dark_colors:
        return 104, 31, 17
    elif color in light_colors:
        return 255, 255, 255
    else:
        return txt_color

def bbox_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area else 0


def draw_detections(image, box, score, class_name, color, unique_count=0):
    x1, y1, x2, y2 = map(int, box)
    label = f"{class_name} {score:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    font_size = min(image.shape[:2]) * 0.0006

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)

    cv2.rectangle(image, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), color, -1)

    cv2.putText(
        image,
        label,
        (x1, y1 - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        get_txt_color(),
        1,
        lineType=cv2.LINE_AA
    )


VID_FORMATS = ['mp4', 'avi', 'mov']
IMG_FORMATS = ['jpg', 'jpeg', 'png']


class LoadMedia:

    def __init__(self, path, img_size=(640, 640)):
        self.img_size = img_size
        self.frames = 0

        if path.isdigit() or path == '0':
            self.type = 'webcam'
            self.cap = cv2.VideoCapture(int(path))
        else:
            file_extension = path.split(".")[-1].lower()
            if file_extension in VID_FORMATS:
                self.type = 'video'
                self.cap = cv2.VideoCapture(path)
                self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            elif file_extension in IMG_FORMATS:
                self.type = 'image'
                self.image = cv2.imread(path)
            else:
                raise ValueError(f"Unsupported format: {path}")

        if self.type == "webcam":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.path = path

    def __iter__(self):
        self.frame = 0  # Resetting the frame count
        return self

    def __next__(self):
        if self.type in ["video", "webcam"]:
            self.cap.grab()
            ret, original_frame = self.cap.retrieve()
            if not ret:
                self.cap.release()
                raise StopIteration
            self.frame += 1
            if self.type == "webcam":
                status = f"{self.type} (frame {self.frame}) Webcam: [{self.path}]: "
            else:
                status = f"{self.type} (frame {self.frame}/{self.frames}) {self.path}: "
        else:
            original_frame = self.image
            if self.frame > 0:
                raise StopIteration
            self.frame += 1
            status = f"{self.type} {self.path}: "

        resized_frame = letterbox(original_frame, self.img_size)[0]  # resize

        return resized_frame, original_frame, status

    def __len__(self):
        return self.frames if self.type == 'video' else 1