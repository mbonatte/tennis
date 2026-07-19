# Analysis options

Scene-cut detection and frame-number rendering require no model. Ball tracking requires `tracknet_model.pt`; bounce detection automatically enables ball tracking and also needs `bounce_model.cbm`. Court detection needs `tennis_court.pt`. Player tracking needs `yolo26n.pt`; pose tracking automatically enables player tracking and needs `yolo26n-pose.pt`. Statistics automatically enable ball and court detection. Point analysis automatically enables scene cuts.

Visual dependencies are server-validated: ball trail → ball tracking; bounce markers → bounce detection; court overlay/keypoints → court detection; player boxes/poses → corresponding player stage; statistics overlay → statistics; history plot → ball tracking. Invalid combinations return HTTP 422.

Statistics, point boundaries, shots, bounces, speeds, distances, and in/out classifications are explicitly experimental. Pose selection uses a job-scoped hybrid box/pose tracker whose ByteTrack state persists across all chunks. TrackNet preserves its two required prior frames at boundaries and performs continuity filtering globally. Court inference is frame-independent and its model persists across chunks; homographies are not smoothed, so difficult footage may still show ordinary frame-to-frame jitter.
