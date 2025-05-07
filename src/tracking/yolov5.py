import cv2
import onnxruntime
import numpy as np

import torch
import torchvision

from typing import Tuple, List


class YOLOv5:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300, nms_mode: str = 'torchvision') -> None:
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode

        self._initialize_model(model_path=model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        outputs = self.inference(image)
        predictions = self.postprocess(outputs)
        return predictions

    def inference(self, image: np.ndarray) -> List[np.ndarray]:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def _initialize_model(self, model_path: str) -> None:
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get model metadata
            metadata = self.session.get_modelmeta().custom_metadata_map
            self.stride = int(metadata.get("stride", 32))  # Default stride value
            self.names = eval(metadata.get("names", "{}"))  # Default to empty dict
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        image = image[::-1]  # Convert BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0  # Normalize the input
        image_tensor = image[np.newaxis, ...]  # Add batch dimension

        return image_tensor

    def postprocess(self, prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = np.squeeze(prediction[0])

        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        classes = outputs[:, 5:]

        boxes = self.xywh2xyxy(boxes)

        # Apply confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        class_ids = np.argmax(classes, axis=1)

        # Filter only "person" class
        person_class_ids = [k for k, v in self.names.items() if v.lower() == 'person']
        person_mask = np.isin(class_ids, person_class_ids)

        boxes = boxes[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]

        # Apply NMS
        if self.nms_mode == "torchvision":
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold).numpy()
        else:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
            indices = [i[0] for i in indices] if len(indices) > 0 else []

        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]
        return boxes, scores, class_ids


    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """xywh -> xyxy

        Args:
            x (np.ndarray): [x, y, w, h]

        Returns:
            np.ndarray: [x1, y1, x2, y2]
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y