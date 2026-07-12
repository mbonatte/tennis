# Model files

Model weights are deliberately excluded from Git and the Docker image. Put them in `MODEL_ROOT` (default `models/`; `/app/models` in Compose). The files below were found in the original local `weights/` directory. Their hashes identify those local copies; they are not a claim of upstream authenticity.

| Filename | Feature | Local size | Observed SHA-256 | Source / license |
|---|---|---:|---|---|
| `tracknet_model.pt` | ball tracking | 42.9 MB | `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076` | Not recorded; manual verification required |
| `bounce_model.cbm` | bounce detection | 0.33 MB | `f525c96b843e47e261a4ea3fbe80f3498980c19821ac41a34b2299a0950ec531` | Not recorded; manual verification required |
| `tennis_court.pt` | court detection/homography | 42.3 MB | `09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04` | Not recorded; manual verification required |
| `yolo26n.pt` | player boxes | 5.5 MB | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Not recorded; likely Ultralytics format, but origin/license must be verified |
| `yolo26n-pose.pt` | player poses | 7.9 MB | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` | Not recorded; likely Ultralytics format, but origin/license must be verified |
| `keypoints_model.pth` | no active code path | 94.6 MB | `16ebb7e46dc88247440c86b388e4f07f0d4abb76ce0a01a22925d3163f7fb7f3` | Not recorded; obsolete unless provenance is established |

None should be redistributed or committed until provenance and redistribution rights are established. The current files exceed ordinary Git best practice; use a private artifact store or Git LFS only after the licensing review.

To copy and verify the known local set:

```bash
python scripts/download_models.py --source weights --destination models
```

No download URLs are embedded because none could be established from the repository.
