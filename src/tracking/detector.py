import onnxruntime as ort
import numpy as np
import cv2
import torch
import torchvision

class YOLOv8ONNX:
    def __init__(self, model_path, input_size=640, conf_thres=0.25, iou_thres=0.45, max_det=300, nms_mode='torchvision'):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode
        self.input_size = input_size

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, ...]  # Add batch dimension
        return img

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        outputs = np.squeeze(outputs[0])

        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        classes = outputs[:, 5:]

        # xywh to xyxy and rescale to original image size
        boxes = self.xywh2xyxy(boxes)
        boxes[:, [0, 2]] *= orig_w / self.input_size
        boxes[:, [1, 3]] *= orig_h / self.input_size

        # Confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        if boxes.shape[0] == 0:
            return []

        class_ids = np.argmax(classes, axis=1)
        # NMS
        if self.nms_mode == "torchvision":
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold).numpy()
        else:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
            else:
                indices = []

        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

        # Format: [x1, y1, x2, y2, conf, class_id]
        detections = []
        for box, score, cls in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = [int(b) for b in box]
            detections.append([x1, y1, x2, y2, float(score), int(cls)])
        return detections