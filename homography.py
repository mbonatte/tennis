from court_reference import CourtReference
import numpy as np
import cv2
from scipy.spatial import distance

court_ref = CourtReference()

# Convert reference keypoints into a NumPy array of shape (N, 1, 2)
# Required format for OpenCV perspectiveTransform
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

# Dictionary to map each court configuration to indices of its keypoints
court_conf_ind = {}

# Loop through all available court configurations
for i in range(len(court_ref.court_conf)):
    
    # Get the configuration (each config contains 4 keypoints)
    conf = court_ref.court_conf[i + 1]
    
    inds = []  # Will store indices of these 4 keypoints
    
    # For each of the 4 keypoints in this configuration
    for j in range(4):
        
        # Find index of this keypoint in the global key_points list
        inds.append(court_ref.key_points.index(conf[j]))
    
    # Store indices for this configuration
    court_conf_ind[i + 1] = inds

def get_trans_matrix(points):
    """
    Determine the best homography matrix from detected court points
    """
    
    matrix_trans = None  # Best transformation matrix (to be found)
    dist_max = np.inf    # Initialize with very large value (we want to minimize error)
    
    # Try all possible 12 court configurations
    for conf_ind in range(1, 13):
        
        # Get the 4 reference points for this configuration
        conf = court_ref.court_conf[conf_ind]

        # Get corresponding indices of these points
        inds = court_conf_ind[conf_ind]
        
        # Extract detected points corresponding to these indices
        inters = [
            points[inds[0]],
            points[inds[1]],
            points[inds[2]],
            points[inds[3]]
        ]
        
        # Only proceed if all 4 required points were detected (no None values)
        if None not in inters:
            
            # Compute homography matrix mapping reference points → detected points
            matrix, _ = cv2.findHomography(
                np.float32(conf),       # Source points (reference court)
                np.float32(inters),     # Destination points (detected)
                method=0                # Direct linear transform (no RANSAC)
            )
            
            # Apply transformation to all reference keypoints
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix).squeeze(1)
            
            dists = []  # Store distances for validation
            
            # Compare transformed keypoints with detected ones
            for i in range(12):
                
                # Skip points used to compute homography and missing points
                if i not in inds and points[i] is not None:
                    
                    # Compute Euclidean distance between predicted and transformed point
                    dists.append(distance.euclidean(points[i], trans_kps[i]))
            
            # Compute average distance (error metric)
            dist_median = np.mean(dists)
            
            # If this configuration gives a better (smaller) error → update best matrix
            if dist_median < dist_max:
                matrix_trans = matrix
                dist_max = dist_median
    
    # Return the best homography matrix found (or None if none valid)
    return matrix_trans
