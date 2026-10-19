"""Download a pinned display detector and export a fixed-shape ONNX model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_BLOBS = {
    "display": "be6cd6c5646b2d0d0390e88f5871bbaf225bc551",
    "digits": "555e491e5ba132cdc9f8d5f29acb1032b0cb82f0",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=MODEL_BLOBS, default="display")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    for key, relative in {
        "YOLO_CONFIG_DIR": ".cache/ultralytics", "TORCH_HOME": ".cache/torch",
        "TEMP": ".tmp", "TMP": ".tmp",
    }.items():
        directory = ROOT / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    url = f"https://raw.githubusercontent.com/adbartx/multimeter-ocr/main/models/{args.kind}.pt"
    weight = models / f"{args.kind}.pt"
    content = weight.read_bytes() if weight.exists() else urllib.request.urlopen(url, timeout=120).read()
    blob_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob_hash != MODEL_BLOBS[args.kind]:
        raise ValueError(f"Unexpected model content: git blob SHA1 {blob_hash}")
    if not weight.exists():
        weight.write_bytes(content)
    manifest = {
        "repository": "https://github.com/adbartx/multimeter-ocr",
        "source_url": url, "git_blob_sha1": blob_hash,
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        "license": "AGPL-3.0 (repository); see upstream LICENSE and README",
    }
    (models / f"{args.kind}.source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.download_only:
        return
    from ultralytics import YOLO

    model = YOLO(str(weight))
    print(f"Task={model.task}; classes={model.names}", flush=True)
    exported = model.export(format="onnx", imgsz=args.size, batch=1, dynamic=False,
                            simplify=False, opset=12, nms=False, device="cpu", half=False)
    print(f"Exported: {exported}", flush=True)


if __name__ == "__main__":
    main()
