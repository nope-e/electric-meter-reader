"""Read LCD digits from largest-area windows with an ONNX digit detector."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from main import ROOT, decode, make_overview, preprocess, read_image, save_image


def format_reading(digits):
    return f"{digits[:-2] or '0'}.{digits[-2:]}" if len(digits) >= 3 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=ROOT / "results/lcd_largest/detections.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/digits.onnx")
    parser.add_argument("--reference", type=Path, default=ROOT / "lcd_reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lcd_recognition")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()
    if not 0 < args.conf < 1 or not 0 < args.iou < 1:
        parser.error("conf/iou must be between zero and one")
    windows = json.loads(args.windows.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
    names = {int(k): v for k, v in names.items()}
    session.run(None, {model_input.name: np.zeros(model_input.shape, np.float32)})
    results, overview = [], []
    for item in windows["images"]:
        path = Path(item["image"])
        label = reference["labels"].get(path.name, {"raw_digits": None, "reason": "unlabeled"})
        result = {"image": str(path), "reference": label, "raw_digits": "", "reading": None,
                  "min_digit_score": None, "mean_digit_score": None, "digit_detections": [],
                  "inference_ms": None, "status": "no_window", "reasons": [], "exact_match": None}
        if not item["detections"]:
            results.append(result)
            continue
        start = time.perf_counter()
        image = read_image(path)
        window = item["detections"][0]
        x1, y1, x2, y2 = window["xyxy"]
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
        crop = image[y1:y2, x1:x2]
        tensor, scale, left, top = preprocess(crop, *model_input.shape[2:])
        infer_start = time.perf_counter()
        output = session.run(None, {model_input.name: tensor})[0]
        result["inference_ms"] = (time.perf_counter() - infer_start) * 1000
        candidates, _ = decode(output, scale, left, top, crop.shape, names, args.conf, args.iou, agnostic=True)
        digits = sorted((d for d in candidates if d["class_name"] in "0123456789" and len(d["class_name"]) == 1),
                        key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) / 2)
        raw = "".join(d["class_name"] for d in digits)
        scores = [d["score"] for d in digits]
        reasons = []
        if x1 <= 0 or y1 <= 0 or x2 >= image.shape[1] or y2 >= image.shape[0]:
            reasons.append("window_touches_image_edge")
        if len(raw) < 3:
            reasons.append("too_few_digits")
        result.update({"window_xyxy": window["xyxy"], "window_score": window["score"],
                       "raw_digits": raw, "reading": format_reading(raw),
                       "min_digit_score": min(scores) if scores else None,
                       "mean_digit_score": float(np.mean(scores)) if scores else None,
                       "digit_detections": digits, "all_symbols": candidates,
                       "status": "review" if reasons else "candidate", "reasons": reasons,
                       "processing_ms": (time.perf_counter() - start) * 1000})
        if label["raw_digits"] is not None:
            result["exact_match"] = raw == label["raw_digits"]
        annotated = crop.copy()
        thickness = max(1, round(crop.shape[1] / 1000))
        for digit in digits:
            a, b, c, d = map(round, digit["xyxy"])
            cv2.rectangle(annotated, (a, b), (c, d), (0, 255, 0), thickness)
            cv2.putText(annotated, f"{digit['class_name']} {digit['score']:.2f}",
                        (a, max(16, b - 4)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, thickness * 0.3),
                        (0, 255, 0), thickness)
        annotated_path = args.output / "annotated" / f"{path.stem}.jpg"
        save_image(annotated_path, annotated)
        truth = label["raw_digits"] or "unscored"
        score = f"{min(scores):.2f}" if scores else "none"
        text = f"pred={raw or 'none'} ref={truth} min={score}"
        overview.append((annotated_path, text))
        result["annotated"] = str(annotated_path)
        results.append(result)
        print(f"{path.name}: {text}", flush=True)
    scored = [r for r in results if r["exact_match"] is not None]
    matched = sum(r["exact_match"] for r in scored)
    summary = {"model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "windows_file": str(args.windows), "reference_file": str(args.reference),
               "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
               "runtime": f"onnxruntime {ort.__version__}", "input_shape": model_input.shape, "class_names": names,
               "confidence_threshold": args.conf, "nms_iou_threshold": args.iou, "class_agnostic_nms": True,
               "image_count": len(results), "labeled_readable_count": len(scored), "exact_matches": matched,
               "exact_match_rate_on_readable_subset": matched / len(scored) if scored else None,
               "unscored_count": len(results) - len(scored),
               "median_digit_inference_ms": float(np.median([r["inference_ms"] for r in results if r["inference_ms"] is not None])),
               "reference_source": reference["source"],
               "note": "Scores are detector scores, not calibrated reading correctness. Candidate output is not auto-accepted. Desktop CPU timings only."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readings.json").write_text(json.dumps({"summary": summary, "images": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["image", "raw_digits", "reading", "min_digit_score", "mean_digit_score", "reference_raw", "exact_match", "status", "reasons", "inference_ms"]
    with (args.output / "readings.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {key: r.get(key) for key in fields}
            row.update(image=Path(r["image"]).name, reference_raw=r["reference"]["raw_digits"], reasons=";".join(r["reasons"]))
            writer.writerow(row)
    if overview:
        make_overview(overview, args.output / "overview.jpg", columns=3, tile_w=800, tile_h=260)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
