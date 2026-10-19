"""Compare pretrained HyperIQA/MUSIQ scores on LCD windows (exploratory only)."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
# Keep every library's download/cache/temporary output inside this experiment.
for key, relative in {
    "TORCH_HOME": ".cache/torch", "HF_HOME": ".cache/huggingface",
    "XDG_CACHE_HOME": ".cache", "TEMP": ".tmp", "TMP": ".tmp",
}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import cv2
import numpy as np
import torch

from main import make_overview, read_image, save_image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_input(image, short_side):
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    # Preserve aspect ratio. Resize the entire LCD rather than square-warping it.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interpolation)


def to_tensor(image):
    rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None])
    return torch.from_numpy(rgb).float().div_(255)


def build_inputs(args):
    windows = json.loads(args.windows.read_text(encoding="utf-8"))
    references = json.loads(args.reference.read_text(encoding="utf-8"))
    readings = json.loads(args.readings.read_text(encoding="utf-8"))
    predictions = {Path(r["image"]).name: r["raw_digits"] for r in readings["images"]}
    rows = []
    for item in windows["images"]:
        if not item["detections"]:
            raise ValueError(f"No LCD window for {item['image']}")
        path = Path(item["image"])
        image = read_image(path)
        box = item["detections"][0]["xyxy"]
        x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
        crop = image[y1:y2, x1:x2]
        resized = resize_input(crop, args.short_side)
        target = args.output / "inputs" / f"{path.stem}.png"
        save_image(target, resized)
        reference = references["labels"].get(path.name, {"raw_digits": None})
        rows.append({
            "id": path.stem, "source_image": str(path), "source_sha256": sha256(path),
            "input": str(target), "input_sha256": sha256(target),
            "input_width": resized.shape[1], "input_height": resized.shape[0],
            "window_xyxy": box, "reference": reference,
            "prediction_digits": predictions.get(path.name), "kind": "real",
            "group": "reference_available" if reference["raw_digits"] is not None else "reference_unconfirmed",
            "scores": {}, "ranks_low_to_high": {}, "inference_ms": {},
        })
    # Known artificial damage, separate from the 30 real photos and their rankings.
    for stem in ["1792403667245", "1792403667695", "1792403668165"]:
        base = read_image(args.output / "inputs" / f"{stem}.png")
        h, w = base.shape[:2]
        occluded = base.copy()
        occluded[round(h * 0.12):round(h * 0.93), round(w * 0.40):round(w * 0.61)] = 255
        variants = {
            "clean": base,
            "blur_sigma8": cv2.GaussianBlur(base, (0, 0), sigmaX=8),
            "opaque_white_mask": occluded,
            "right_quarter_cut": base[:, :round(w * 0.75)],
        }
        for name, image in variants.items():
            target = args.output / "controls" / f"{stem}_{name}.png"
            save_image(target, image)
            rows.append({
                "id": f"{stem}_{name}", "base_id": stem, "variant": name,
                "input": str(target), "input_sha256": sha256(target),
                "input_width": image.shape[1], "input_height": image.shape[0],
                "kind": "synthetic_control", "group": name,
                "scores": {}, "ranks_low_to_high": {}, "inference_ms": {},
            })
    return rows


def load_metric(name):
    import pyiqa

    metric = pyiqa.create_metric(name, device="cpu").eval()
    config = {"implementation": f"pyiqa {version('pyiqa')}", "architecture": type(metric.net).__name__,
              "parameters": sum(p.numel() for p in metric.net.parameters()),
              "lower_better": metric.lower_better,
              "hyperiqa_num_crop": getattr(metric.net, "num_crop", None),
              "training_dataset": "KonIQ-10k (pyiqa default pretrained model)"}
    return metric, config


def verify_or_download_weights(name, allow_download=False):
    filenames = {
        "hyperiqa": "HyperIQA-resnet50-koniq10k-c96c41b1.pth",
        "musiq": "musiq_koniq_ckpt-e95806b9.pth",
    }
    filename = filenames[name]
    prefix = filename.rsplit("-", 1)[1].split(".")[0]
    target = ROOT / ".cache/torch/hub/pyiqa" / filename
    if target.exists() and sha256(target).startswith(prefix):
        return
    if not allow_download:
        raise ValueError(f"Missing/corrupt weight {target}; run --download-only first")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    sources = [
        f"https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/{filename}",
        f"https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/{filename}",
    ]
    errors = []
    for url in sources:
        print(f"Downloading verified {name} weights: {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as f:
                size = 0
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
                    size += len(chunk)
                    if size % (16 * 1024 * 1024) == 0:
                        print(f"  {size // (1024 * 1024)} MiB", flush=True)
            digest = sha256(partial)
            if not digest.startswith(prefix):
                raise ValueError(f"SHA256 {digest} does not match published prefix {prefix}")
            partial.replace(target)
            print(f"Verified {filename}: {digest}", flush=True)
            return
        except Exception as error:
            errors.append(f"{url}: {error}")
            print(errors[-1], flush=True)
    raise RuntimeError("Could not obtain verified model weights: " + "; ".join(errors))


def evaluate(rows, names):
    real = [r for r in rows if r["kind"] == "real"]
    summary = {}
    for name in names:
        ordered = sorted(real, key=lambda r: (r["scores"][name], r["id"]))
        for rank, row in enumerate(ordered, 1):
            row["ranks_low_to_high"][name] = rank
        flagged = [r for r in real if r["group"] == "reference_unconfirmed"]
        available = [r for r in real if r["group"] == "reference_available"]
        wins = sum((a["scores"][name] < b["scores"][name]) +
                   0.5 * (a["scores"][name] == b["scores"][name]) for a in flagged for b in available)
        lowest = ordered[:len(flagged)]
        control_comparisons = []
        for row in rows:
            if row["kind"] != "synthetic_control" or row["variant"] == "clean":
                continue
            base = next(r for r in rows if r["id"] == f"{row['base_id']}_clean")
            control_comparisons.append({"id": row["id"], "variant": row["variant"],
                                        "score": row["scores"][name], "clean_score": base["scores"][name],
                                        "damaged_scores_lower": row["scores"][name] < base["scores"][name]})
        summary[name] = {
            "score_direction": "higher is better; not a reading correctness probability",
            "median_inference_ms": float(np.median([r["inference_ms"][name] for r in real])),
            "lowest_five_ids": [r["id"] for r in lowest],
            "unconfirmed_in_lowest_five": sum(r["group"] == "reference_unconfirmed" for r in lowest),
            "unconfirmed_ranks": {r["id"]: r["ranks_low_to_high"][name] for r in flagged},
            "exploratory_auc_unconfirmed_vs_available": wins / (len(flagged) * len(available)),
            "auc_note": "Reference unconfirmed is an assistant review flag, NOT unreadability ground truth. This is ranking exploration, NOT classifier validation.",
            "controls": control_comparisons,
        }
    return summary


def write_results(args, rows, metadata):
    summary = evaluate(rows, args.models)
    weights = []
    for cache in [ROOT / ".cache/torch", ROOT / ".cache/huggingface"]:
        for path in cache.rglob("*"):
            if path.is_file() and path.stat().st_size > 10_000_000:
                weights.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "sha256": sha256(path)})
    result = {"configuration": {"models": metadata, "torch": torch.__version__, "device": "cpu",
                                "threads": args.threads, "short_side": args.short_side,
                                "preprocessing": "Original-image LCD crop, aspect-preserving resize short side to 256 by default, RGB float32 NCHW /255. Model-specific normalization/crops handled by pyiqa.",
                                "window_file_sha256": sha256(args.windows), "reference_sha256": sha256(args.reference),
                                "note": "Exploratory IQA, not calibrated LCD readability. No recognition predictions or references are passed to IQA models. PyTorch only; ONNX/mobile not tested.",
                                "weights": weights}, "summary": summary, "images": rows}
    (args.output / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["id", "kind", "group", "reference_raw", "prediction_digits"]
    fields += [f"{n}_{k}" for n in args.models for k in ["score", "rank", "ms"]]
    with (args.output / "quality.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {k: row.get(k) for k in fields[:3]}
            record.update(reference_raw=row.get("reference", {}).get("raw_digits"),
                          prediction_digits=row.get("prediction_digits"))
            for name in args.models:
                record.update({f"{name}_score": row["scores"][name],
                               f"{name}_rank": row["ranks_low_to_high"].get(name),
                               f"{name}_ms": row["inference_ms"][name]})
            writer.writerow(record)
    real = [r for r in rows if r["kind"] == "real"]
    for name in args.models:
        ordered = sorted(real, key=lambda r: r["ranks_low_to_high"][name])
        items = [(Path(r["input"]), f"rank={r['ranks_low_to_high'][name]} {name}={r['scores'][name]:.4f} "
                  f"{'REF?' if r['group'] == 'reference_unconfirmed' else 'ref available'}") for r in ordered]
        make_overview(items, args.output / f"{name}_ranked.jpg", columns=3, tile_w=800, tile_h=260)
    controls = [r for r in rows if r["kind"] == "synthetic_control"]
    make_overview([(Path(r["input"]), " ".join(f"{n}={r['scores'][n]:.4f}" for n in args.models))
                   for r in controls], args.output / "controls_overview.jpg", columns=2, tile_w=1000, tile_h=300)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["hyperiqa", "musiq"], default=["hyperiqa", "musiq"])
    parser.add_argument("--windows", type=Path, default=ROOT / "results/lcd_largest/detections.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "lcd_reference.json")
    parser.add_argument("--readings", type=Path, default=ROOT / "results/lcd_recognition/readings.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lcd_quality")
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.short_side < 224 or args.threads < 1:
        parser.error("short-side must be >=224 and threads >=1")
    args.models = list(dict.fromkeys(args.models))
    if not args.download_only:
        # The timm backbone is already cached after the preparation step.
        # Prevent needless online metadata checks during actual CPU benchmarking.
        os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [] if args.download_only else build_inputs(args)
    metadata = {}
    for name in args.models:
        print(f"Loading {name}...", flush=True)
        verify_or_download_weights(name, args.download_only)
        metric, config = load_metric(name)
        metadata[name] = config
        if args.download_only:
            print(json.dumps(config), flush=True)
            continue
        with torch.inference_mode():
            metric(to_tensor(read_image(Path(rows[0]["input"]))))  # warmup, not timed
            for row in rows:
                tensor = to_tensor(read_image(Path(row["input"])))
                start = time.perf_counter()
                score = float(metric(tensor).item())
                row["inference_ms"][name] = (time.perf_counter() - start) * 1000
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite {name} score for {row['id']}")
                row["scores"][name] = score
                print(f"{name}: {row['id']} score={score:.5f} {row['inference_ms'][name]:.1f}ms", flush=True)
        del metric
    if not args.download_only:
        write_results(args, rows, metadata)


if __name__ == "__main__":
    main()
