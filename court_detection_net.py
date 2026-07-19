import cv2
import numpy as np
import torch

from homography import get_trans_matrix, refer_kps  # Homography utilities
from postprocess import refine_kps  # Custom function to refine keypoint locations
from tennis_analyzer.errors import VideoProcessingError
from tracknet import Tracker  # Custom neural network model for keypoint tracking

class CourtDetectorNet:
    def __init__(self, path_model=None, device="cpu", model=None):
        # Initialize the neural network with 15 output channels (keypoints/heatmaps)
        self.model = model or Tracker(out_channels=15)
        self.device = torch.device(device)
        if path_model:
            try:
                checkpoint = torch.load(path_model, map_location=self.device, weights_only=True)
                state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                self.model.load_state_dict(state_dict)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise VideoProcessingError("The court-detection checkpoint is incompatible or unreadable") from exc
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer_model(self, frames):
        if not frames:
            return [], []
        # Define output resolution for resizing input frames
        output_width = 640
        output_height = 360

        orig_h, orig_w = frames[0].shape[:2]

        # Scale factor used later to map predictions back to original resolution
        scale_x = orig_w / output_width
        scale_y = orig_h / output_height
        
        # Lists to store results
        kps_res = []        # Keypoints results
        matrixes_res = []   # Transformation matrices

        # Loop over each frame
        for image in frames:
            # Resize image to model input size
            img = cv2.resize(image, (output_width, output_height))
            
            # Normalize pixel values to [0, 1]
            inp = (img.astype(np.float32) / 255.)
            
            # Change image shape from (H, W, C) to (C, H, W)
            inp = torch.from_numpy(np.rollaxis(inp, 2, 0))
            
            # Add batch dimension → shape becomes (1, C, H, W)
            inp = inp.unsqueeze(0)

            # Run inference through the model
            out = self.model(inp.to(self.device, dtype=torch.float32))[0]
            
            # Apply sigmoid to convert outputs to probabilities
            pred = torch.sigmoid(out).cpu().numpy()

            # Store detected keypoints for this frame
            points = []

            # Loop through first 14 keypoint heatmaps (ignore last channel)
            for kps_num in range(14):
                # Convert heatmap to 0–255 grayscale image
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                
                # Apply binary threshold to highlight strong responses
                low_thresh = 170
                ret, heatmap = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
                
                # Detect circular blobs (keypoints) using Hough Circle Transform
                circles = cv2.HoughCircles(
                    heatmap,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=2,
                    minRadius=10,
                    maxRadius=25
                )

                # If a circle (keypoint) is detected
                if circles is not None:
                    orig_h, orig_w = image.shape[:2]
                    scale_x = orig_w / output_width
                    scale_y = orig_h / output_height
                    
                    # Extract predicted x and y coordinates and scale them
                    x_pred = circles[0][0][0] * scale_x
                    y_pred = circles[0][0][1] * scale_y

                    # Refine keypoints for most indices (skip some specific ones)
                    if kps_num not in [8, 12, 9] and x_pred and y_pred:
                        x_pred, y_pred = refine_kps(
                            image,
                            int(y_pred),
                            int(x_pred),
                            crop_size=40
                        )

                    # Store detected point as (x, y)
                    points.append((x_pred, y_pred))
                
                else:
                    # If no keypoint detected, store None
                    points.append(None)

            use_homography = True
            if use_homography:
                # Compute homography transformation matrix from detected points
                matrix_trans = get_trans_matrix(points) 
            
                # Reset points (will store transformed reference points instead)
                points = None
                
                # If a valid transformation matrix was found
                if matrix_trans is not None:
                    # Transform reference keypoints into image space
                    points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                    
                    # Invert the transformation matrix
                    matrix_trans = cv2.invert(matrix_trans)[1]

                matrixes_res.append(matrix_trans)
            
            # Store results for this frame
            kps_res.append(points)
            
            
        # Return all transformation matrices and keypoints
        return matrixes_res, kps_res
