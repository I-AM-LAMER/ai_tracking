import cv2
import numpy as np

def letterbox(image, target_shape=(640, 640), color=(114, 114, 114)):
    height, width = image.shape[:2]
    scale = min(target_shape[0] / height, target_shape[1] / width)
    new_size = (int(width * scale), int(height * scale))
    image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_shape[1] - new_size[0]) / 2, (target_shape[0] - new_size[1]) / 2
    top, bottom = int(dh), int(target_shape[0] - new_size[1] - int(dh))
    left, right = int(dw), int(target_shape[1] - new_size[0] - int(dw))
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_padded, scale, (dw, dh)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`.
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
    return boxes