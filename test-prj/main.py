"""Batch-test LCD window detection using ONNX Runtime, without PyTorch at runtime."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def preprocess(image, height, width):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized_w, resized_h = round(w * scale), round(h * scale)
    left = round((width - resized_w) / 2 - 0.1)
    top = round((height - resized_h) / 2 - 0.1)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    # Rounding resize dimensions gives slightly different effective X/Y scales.
    return tensor, (resized_w / w, resized_h / h), left, top


def decode(output, scale, left, top, image_shape, class_names, confidence, iou, agnostic=False):
    # Standard Ultralytics detection export, nms=False: [1, 4 + classes, anchors].
    expected = 4 + len(class_names)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected:
        raise ValueError(f"Expected [1, {expected}, anchors], got {output.shape}")
    rows = output[0].T
    scores = rows[:, 4:].max(axis=1)
    best_raw_score = float(scores.max())
    rows = rows[scores >= confidence]
    if len(rows) == 0:
        return [], best_raw_score
    scores = rows[:, 4:].max(axis=1)
    classes = rows[:, 4:].argmax(axis=1)
    xywh = rows[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    kept = []
    groups = [np.arange(len(rows))] if agnostic else [np.flatnonzero(classes == c) for c in np.unique(classes)]
    for indices in groups:
        selected = cv2.dnn.NMSBoxes(
            xywh[indices].tolist(), scores[indices].tolist(), confidence, iou
        )
        kept.extend(indices[np.asarray(selected, dtype=int).reshape(-1)].tolist())
    h, w = image_shape[:2]
    scale_x, scale_y = scale
    detections = []
    for index in sorted(kept, key=lambda i: float(scores[i]), reverse=True):
        x, y, bw, bh = xywh[index]
        box = [
            max(0.0, min(float(w), float((x - left) / scale_x))),
            max(0.0, min(float(h), float((y - top) / scale_y))),
            max(0.0, min(float(w), float((x + bw - left) / scale_x))),
            max(0.0, min(float(h), float((y + bh - top) / scale_y))),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        class_id = int(classes[index])
        detections.append({
            "xyxy": box, "score": float(scores[index]),
            "class_id": class_id, "class_name": class_names[class_id],
        })
    return detections, best_raw_score


def make_overview(items, output_path, columns=4, tile_w=480, tile_h=400):
    canvas = np.full((math.ceil(len(items) / columns) * tile_h, columns * tile_w, 3), 30, np.uint8)
    for index, (path, label) in enumerate(items):
        image = read_image(path)
        scale = min(tile_w / image.shape[1], (tile_h - 50) / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        offset = (tile_w - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x + offset:x + offset + resized.shape[1]] = resized
        cv2.putText(canvas, path.stem, (x + 8, y + tile_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        cv2.putText(canvas, label, (x + 8, y + tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 230, 230), 1)
    save_image(output_path, canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/display.onnx")
    parser.add_argument("--input", type=Path, default=ROOT.parent / "datas/lcd")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lcd_largest")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.conf < 1 or not 0 < args.iou < 1 or args.threads < 1:
        parser.error("conf/iou must be between 0 and 1; threads must be positive")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error(f"No input images: {args.input}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    start = time.perf_counter()
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - start) * 1000
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(n, int) for n in shape):
        raise ValueError(f"This runner needs fixed NCHW input [1, 3, H, W], got {shape}")
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): value for key, value in names.items()}
    session.run(None, {model_input.name: np.zeros(shape, dtype=np.float32)})
    results, overview = [], []
    for path in paths:
        start = time.perf_counter()
        image = read_image(path)
        tensor, scale, left, top = preprocess(image, shape[2], shape[3])
        infer_start = time.perf_counter()
        output = session.run(None, {model_input.name: tensor})[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000
        candidates, best_raw = decode(output, scale, left, top, image.shape, names, args.conf, args.iou)
        # Choose by area after mapping/clipping to the original image; score breaks ties.
        detections = [max(candidates, key=lambda d: (
            (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]),
            d["score"],
        ))] if candidates else []
        process_ms = (time.perf_counter() - start) * 1000
        annotated = image.copy()
        thickness = max(2, round(max(image.shape[:2]) / 1000))
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["xyxy"]
            x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            label = f"{detection['class_name']} {detection['score']:.3f}"
            cv2.putText(annotated, label, (x1, max(25, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, thickness * 0.35), (0, 255, 0), thickness)
            crop_path = args.output / "crops" / f"{path.stem}_{index:02d}.jpg"
            save_image(crop_path, image[y1:y2, x1:x2])
            detection["crop"] = str(crop_path.resolve())
        annotated_path = args.output / "annotated" / f"{path.stem}.jpg"
        save_image(annotated_path, annotated)
        selected_score = f"{detections[0]['score']:.3f}" if detections else "none"
        label = f"candidates={len(candidates)} selected={selected_score} ORT={infer_ms:.1f}ms"
        overview.append((annotated_path, label))
        results.append({
            "image": str(path.resolve()), "width": image.shape[1], "height": image.shape[0],
            "detections": detections, "candidate_count": len(candidates), "best_raw_score": best_raw,
            "inference_ms": infer_ms, "processing_ms": process_ms,
            "annotated": str(annotated_path.resolve()),
        })
        print(f"{path.name}: {label}", flush=True)
    summary = {
        "model": str(args.model.resolve()), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "runtime": f"onnxruntime {ort.__version__}", "providers": session.get_providers(),
        "input_shape": shape, "class_names": names, "confidence_threshold": args.conf,
        "nms_iou_threshold": args.iou, "threads": args.threads,
        "selection": "largest_area_in_original_image_score_tiebreak",
        "candidate_count_before_selection": sum(r["candidate_count"] for r in results),
        "image_count": len(results), "images_with_detections": sum(bool(r["detections"]) for r in results),
        "detection_count": sum(len(r["detections"]) for r in results),
        "images_with_multiple_detections": sum(len(r["detections"]) > 1 for r in results),
        "session_load_ms": load_ms,
        "median_inference_ms": float(np.median([r["inference_ms"] for r in results])),
        "median_processing_ms": float(np.median([r["processing_ms"] for r in results])),
        "note": "Detection count is not accuracy. No ground-truth boxes; desktop CPU timing excludes output-file writes and is not phone timing.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "detections.json").write_text(json.dumps({"summary": summary, "images": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    make_overview(overview, args.output / "overview.jpg")
    top_windows = [
        (Path(r["detections"][0]["crop"]), f"largest area; score={r['detections'][0]['score']:.3f}")
        for r in results if r["detections"]
    ]
    if top_windows:
        make_overview(top_windows, args.output / "top_windows.jpg", columns=3, tile_w=640, tile_h=220)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
