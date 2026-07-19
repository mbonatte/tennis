"""User-supplied static-camera court calibration helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from court_reference import CourtReference

CALIBRATION_SCHEMA_VERSION = 1
CORNER_LABELS = (
    "Far left doubles corner",
    "Far right doubles corner",
    "Near right doubles corner",
    "Near left doubles corner",
)


class CourtCalibrationError(ValueError):
    """A user-supplied court calibration cannot produce a valid homography."""


def reference_corners() -> np.ndarray:
    """Return the four outer-court corners in the required click order."""
    return np.asarray(CourtReference().border_points, dtype=np.float32)


def create_static_calibration(frame_index: int, image_points: Any) -> dict[str, Any]:
    """Create an image-to-court calibration from four ordered image points."""
    if not isinstance(frame_index, int) or frame_index < 0:
        raise CourtCalibrationError("Calibration frame must be a non-negative integer")
    points = _validated_points(image_points)
    if abs(cv2.contourArea(points)) < 1.0 or not cv2.isContourConvex(points.astype(np.int32)):
        raise CourtCalibrationError("Court corners must form a non-degenerate convex quadrilateral")
    homography = cv2.getPerspectiveTransform(points, reference_corners())
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        raise CourtCalibrationError("Court corners cannot form a valid perspective transform")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "mode": "static",
        "frame_index": frame_index,
        "image_points": points.tolist(),
    }


def calibration_homography(calibration: dict[str, Any]) -> np.ndarray:
    """Return the image-to-reference homography for a saved calibration."""
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION or calibration.get("mode") != "static":
        raise CourtCalibrationError("Unsupported court calibration")
    return cv2.getPerspectiveTransform(_validated_points(calibration.get("image_points")), reference_corners())


def calibrated_keypoints(calibration: dict[str, Any]) -> np.ndarray:
    """Project all reference-court keypoints into the calibrated source frame."""
    inverse = np.linalg.inv(calibration_homography(calibration)).astype(np.float32)
    reference = np.asarray(CourtReference().key_points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(reference, inverse)


def suggested_outer_corners(keypoints: Any) -> list[list[float]] | None:
    """Convert model keypoints to the user-facing outer-corner click order."""
    if not isinstance(keypoints, list) or len(keypoints) < 4:
        return None
    try:
        points = np.asarray([keypoints[0], keypoints[1], keypoints[3], keypoints[2]], dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        return None
    return points.tolist() if np.isfinite(points).all() else None


def _validated_points(image_points: Any) -> np.ndarray:
    try:
        points = np.asarray(image_points, dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError) as exc:
        raise CourtCalibrationError("Exactly four court-corner points are required") from exc
    if not np.isfinite(points).all():
        raise CourtCalibrationError("Court-corner points must be finite numbers")
    return points
