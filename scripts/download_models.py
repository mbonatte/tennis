from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

KNOWN = {
    "tracknet_model.pt": "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076",
    "bounce_model.cbm": "f525c96b843e47e261a4ea3fbe80f3498980c19821ac41a34b2299a0950ec531",
    "tennis_court.pt": "09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "yolo26n-pose.pt": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify manually obtained model files")
    parser.add_argument("--source", type=Path, required=True, help="Directory containing legally obtained weights")
    parser.add_argument("--destination", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, expected in KNOWN.items():
        source = args.source / filename
        if not source.is_file():
            missing.append(filename)
            continue
        actual = digest(source)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch for {filename}: {actual}")
        shutil.copy2(source, args.destination / filename)
        print(f"installed {filename}")  # CLI output, not application logging
    if missing:
        raise SystemExit("Missing model file(s): " + ", ".join(missing))


if __name__ == "__main__":
    main()
