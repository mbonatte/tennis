import cv2
import numpy as np
import torch

from homography import get_trans_matrix, refer_kps
from postprocess import refine_kps
from tennis_analyzer.checkpoints import normalize_state_dict
from tennis_analyzer.errors import VideoProcessingError
from tracknet import Tracker


class CourtDetectorNet:
    def __init__(self, path_model=None, device="cpu", model=None):
        self.model = model or Tracker(out_channels=15)
        self.device = torch.device(device)
        if path_model:
            try:
                checkpoint = torch.load(path_model, map_location=self.device, weights_only=True)
                self.model.load_state_dict(normalize_state_dict(checkpoint))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise VideoProcessingError("The court-detection checkpoint is incompatible or unreadable") from exc
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer_model(self, frames, batch_size=1):
        if not frames:
            return [], []
        if batch_size <= 0:
            raise ValueError("court batch size must be positive")
        matrices, keypoints = [], []
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            inputs = np.stack([np.moveaxis(cv2.resize(frame, (640, 360)), 2, 0) for frame in batch]).astype(np.float32)
            predictions = torch.sigmoid(self.model(torch.from_numpy(inputs / 255.0).to(self.device))).cpu().numpy()
            for image, prediction in zip(batch, predictions, strict=True):
                points = self._points(image, prediction)
                matrix = get_trans_matrix(points)
                projected = None
                if matrix is not None:
                    projected = cv2.perspectiveTransform(refer_kps, matrix)
                    matrix = cv2.invert(matrix)[1]
                matrices.append(matrix)
                keypoints.append(projected)
        return matrices, keypoints

    @staticmethod
    def _points(image, prediction):
        height, width = image.shape[:2]
        points = []
        for index in range(14):
            heatmap = (prediction[index] * 255).astype(np.uint8)
            _, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(
                heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25
            )
            if circles is None:
                points.append(None)
                continue
            x = circles[0][0][0] * width / 640
            y = circles[0][0][1] * height / 360
            if index not in [8, 9, 12] and x and y:
                x, y = refine_kps(image, int(y), int(x), crop_size=40)
            points.append((x, y))
        return points
