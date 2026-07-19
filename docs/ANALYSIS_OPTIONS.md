# Analysis options

Web uploads always perform scene, ball, court, player, pose, bounce, and statistics analysis so later renders can enable any supported overlay without rerunning inference. The CLI retains selective analysis for diagnostics and development. Rendering options are chosen after web analysis completes; each render is a separate output backed by the saved versioned artifact.

Visual dependencies are server-validated: ball trail → ball tracking; bounce markers → bounce detection; court overlay/keypoints → court detection; player boxes/poses → corresponding player stage; statistics overlay → statistics; history plot → ball tracking. Invalid combinations return HTTP 422.

Statistics, point boundaries, shots, bounces, speeds, distances, and in/out classifications are explicitly experimental. Pose selection uses a job-scoped hybrid box/pose tracker whose ByteTrack state persists across all chunks. TrackNet preserves its two required prior frames at boundaries and performs continuity filtering globally. Court inference is frame-independent and its model persists across chunks; homographies are not smoothed, so difficult footage may still show ordinary frame-to-frame jitter.
