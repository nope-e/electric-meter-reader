"""Classify whole meter photographs as LCD or mechanical wheel using ONNX Runtime."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from main import ROOT, make_overview, read_image, save_image


def preprocess_meter(image):
    """Match the source MobileCLIP processor; image is a PIL RGB image."""
    width, height = image.size
    if width <= height:
        resized = (256, int(height * 256 / width))
    else:
        resized = (int(width * 256 / height), 256)
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left, top = (image.width - 256) // 2, (image.height - 256) // 2
    image = image.crop((left, top, left + 256, top + 256))
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), image


class MeterTypeClassifier:
    def __init__(self, model, threads=4, min_cosine_gap=0.0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.names = json.loads(self.session.get_modelmeta().custom_metadata_map["class_names"])
        self.min_cosine_gap = min_cosine_gap
        self.session.run(None, {self.input_name: np.zeros((1, 3, 256, 256), np.float32)})

    def classify(self, image):
        """No filename, directory label, or reference is used for inference."""
        started = time.perf_counter()
        tensor, processed = preprocess_meter(image)
        infer_start = time.perf_counter()
        cosine, relative_scores = self.session.run(None, {self.input_name: tensor})
        inference_ms = (time.perf_counter() - infer_start) * 1000
        cosine, relative_scores = cosine[0], relative_scores[0]
        order = np.argsort(cosine)[::-1]
        top = int(order[0])
        gap = float(cosine[top] - cosine[order[1]])
        result = {
            "meter_type": self.names[top] if gap > self.min_cosine_gap else "unknown",
            "top_class": self.names[top], "cosine_gap": gap,
            "cosine_scores": dict(zip(self.names, map(float, cosine))),
            "relative_class_scores": dict(zip(self.names, map(float, relative_scores))),
            "inference_ms": inference_ms, "processing_ms": (time.perf_counter() - started) * 1000,
        }
        return result, processed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/mobileclip_s0/meter_type_fp32.onnx")
    parser.add_argument("--input", type=Path, default=ROOT.parent / "datas")
    parser.add_argument("--output", type=Path, default=ROOT / "results/meter_type_fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-cosine-gap", type=float, default=0.0)
    args = parser.parse_args()
    if args.threads < 1 or args.min_cosine_gap < 0:
        parser.error("threads must be >=1 and min-cosine-gap >=0")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error(f"No input images: {args.input}")
    classifier = MeterTypeClassifier(args.model, args.threads, args.min_cosine_gap)
    rows, overview = [], []
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result, processed = classifier.classify(image)
        # Folder names are read only AFTER inference, for evaluation/reporting.
        expected = path.parent.name if path.parent.name in classifier.names else None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_id = f"{path.stem}_{digest[:8]}"
        input_path = args.output / "inputs" / f"{artifact_id}.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(input_path)
        annotated = np.array(image)[:, :, ::-1].copy()
        scale = min(1.0, 1000 / max(annotated.shape[:2]))
        annotated = cv2.resize(annotated, (round(annotated.shape[1] * scale), round(annotated.shape[0] * scale)))
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 55), (20, 20, 20), -1)
        cv2.putText(annotated, f"type={result['meter_type']} gap={result['cosine_gap']:.4f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 240, 100), 2)
        annotated_path = args.output / "annotated" / f"{artifact_id}.jpg"
        save_image(annotated_path, annotated)
        row = {"image": str(path.resolve()), "source_sha256": digest, "reference_type": expected,
               **result, "exact_match": result["meter_type"] == expected if expected else None,
               "model_input_image": str(input_path), "annotated": str(annotated_path)}
        rows.append(row)
        overview.append((annotated_path, f"pred={result['meter_type']} ref={expected or '?'} gap={result['cosine_gap']:.4f}"))
        print(f"{path.name}: {overview[-1][1]} {result['inference_ms']:.1f}ms", flush=True)
    scored = [r for r in rows if r["exact_match"] is not None]
    matched = sum(r["exact_match"] for r in scored)
    summary = {
        "model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "runtime": f"onnxruntime {ort.__version__}", "providers": classifier.session.get_providers(),
        "threads": args.threads, "class_names": classifier.names,
        "input_shape": [1, 3, 256, 256], "min_cosine_gap": args.min_cosine_gap,
        "image_count": len(rows), "labeled_count": len(scored), "exact_matches": matched,
        "accuracy_on_labeled_samples": matched / len(scored) if scored else None,
        "by_reference_type": {name: {
            "count": sum(r["reference_type"] == name for r in scored),
            "matched": sum(r["reference_type"] == name and r["exact_match"] for r in scored),
            "predicted_counts": {label: sum(r["reference_type"] == name and r["meter_type"] == label for r in scored)
                                 for label in [*classifier.names, "unknown"]},
        } for name in classifier.names},
        "median_inference_ms": float(np.median([r["inference_ms"] for r in rows])),
        "median_processing_ms": float(np.median([r["processing_ms"] for r in rows])),
        "note": "Whole-image, closed-set zero-shot classification. Folder names used ONLY for evaluation. Relative scores are not calibrated correctness probabilities. Same small sample set, not independent validation; no phone testing.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "classifications.json").write_text(json.dumps({"summary": summary, "images": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "classifications.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["image", "reference_type", "meter_type", "exact_match", "cosine_gap", "lcd_cosine", "wheel_cosine", "inference_ms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {k: row.get(k) for k in fields}
            record.update(lcd_cosine=row["cosine_scores"]["lcd"], wheel_cosine=row["cosine_scores"]["wheel"])
            writer.writerow(record)
    make_overview(overview, args.output / "overview.jpg", columns=4, tile_w=420, tile_h=500)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
