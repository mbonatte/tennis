import cv2
import numpy as np
import pytest

from tennis_analyzer.pipeline.court_calibration import (
    CourtCalibrationError,
    calibrated_keypoints,
    calibration_homography,
    create_static_calibration,
    reference_corners,
    suggested_outer_corners,
)


def test_static_calibration_maps_clicked_image_corners_to_reference_corners():
    image_points = [[100, 80], [540, 90], [600, 420], [40, 430]]
    calibration = create_static_calibration(12, image_points)

    transformed = cv2.perspectiveTransform(
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2), calibration_homography(calibration)
    )

    assert calibration["frame_index"] == 12
    assert np.allclose(transformed.reshape(4, 2), reference_corners())
    assert calibrated_keypoints(calibration).shape == (14, 1, 2)


@pytest.mark.parametrize("points", [[[1, 2], [3, 4], [5, 6]], [[0, 0], [10, 0], [20, 0], [30, 0]]])
def test_static_calibration_rejects_invalid_corner_sets(points):
    with pytest.raises(CourtCalibrationError):
        create_static_calibration(0, points)


def test_suggested_outer_corners_reorders_model_keypoints():
    keypoints = [[[10, 11]], [[20, 21]], [[30, 31]], [[40, 41]]]

    assert suggested_outer_corners(keypoints) == [[10.0, 11.0], [20.0, 21.0], [40.0, 41.0], [30.0, 31.0]]
