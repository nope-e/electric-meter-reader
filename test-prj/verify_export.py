"""Check ONNX output and postprocessing against the source PyTorch model."""

import argparse
import json
import os

from main import ROOT, decode, preprocess, read_image

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["display", "digits"], default="display")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = YOLO(str(ROOT / f"models/{args.kind}.pt"))
    source.model.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(ROOT / f"models/{args.kind}.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_dir = ROOT.parent / "datas/lcd" if args.kind == "display" else ROOT / "results/lcd_largest/crops"
    paths = sorted(input_dir.glob("*.jpg"))
    selected = [paths[0], paths[len(paths) // 2], paths[-1]]
    checks = []
    for path in selected:
        image = read_image(path)
        tensor, scale, left, top = preprocess(image, *model_input.shape[2:])
        with torch.inference_mode():
            torch_output = source.model(torch.from_numpy(tensor))
            if isinstance(torch_output, (list, tuple)):
                torch_output = torch_output[0]
            torch_output = torch_output.numpy()
        ort_output = session.run(None, {model_input.name: tensor})[0]
        np.testing.assert_allclose(ort_output, torch_output, rtol=1e-3, atol=2e-3)
        detections, _ = decode(ort_output, scale, left, top, image.shape, source.names, 0.25, 0.45,
                               agnostic=args.kind == "digits")
        reference = source.predict(image, imgsz=model_input.shape[2:], rect=False,
                                   conf=0.25, iou=0.45, device="cpu", verbose=False,
                                   agnostic_nms=args.kind == "digits")[0]
        assert len(detections) == len(reference.boxes), (path.name, len(detections), len(reference.boxes))
        if detections:
            np.testing.assert_allclose(
                np.array([d["xyxy"] for d in detections]), reference.boxes.xyxy.numpy(),
                rtol=1e-3, atol=0.1,
            )
        checks.append({
            "image": path.name, "raw_output_max_abs_error": float(np.abs(ort_output - torch_output).max()),
            "reference_box_count": len(reference.boxes), "onnx_box_count": len(detections),
            "status": "passed",
        })
    output = ROOT / "results" / ("export_verification.json" if args.kind == "display" else "digits_export_verification.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
