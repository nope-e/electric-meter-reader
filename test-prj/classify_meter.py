"""Classify whole meter photos as LCD or wheel with the trained ONNX model."""

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

from main import ROOT, make_overview, save_image


def preprocess_meter(image):
    """PIL RGB full image -> fixed NCHW tensor; identical to training."""
    image = ImageOps.pad(image.convert('RGB'), (256, 256),
                         method=Image.Resampling.BILINEAR, color=(128, 128, 128))
    values = np.asarray(image, dtype=np.float32) / np.float32(255)
    values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None]), image


class MeterTypeClassifier:
    def __init__(self, model=ROOT / 'models/meter_type_cnn/meter_type.onnx', threads=4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get('preprocess_kind') != 'rgb_imagenet_pad256':
            raise ValueError('Expected the supervised MobileNet meter classifier and its preprocessing metadata')
        self.names = json.loads(metadata['class_names'])
        self.input_name = self.session.get_inputs()[0].name
        self.session.run(None, {self.input_name: np.zeros((1, 3, 256, 256), dtype=np.float32)})

    def classify(self, image):
        """Accept PIL RGB pixels only; filenames and labels do not enter prediction."""
        started = time.perf_counter()
        tensor, processed = preprocess_meter(image)
        infer_start = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        inference_ms = (time.perf_counter() - infer_start) * 1000
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        top = int(scores.argmax())
        return {
            'meter_type': self.names[top],
            'relative_class_scores': dict(zip(self.names, map(float, scores))),
            'inference_ms': inference_ms,
            'processing_ms': (time.perf_counter() - started) * 1000,
        }, processed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/meter_type_cnn/meter_type.onnx')
    parser.add_argument('--input', type=Path, default=ROOT.parent / 'datas')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/meter_type')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted(p for p in args.input.rglob('*') if p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error(f'No images: {args.input}')
    classifier = MeterTypeClassifier(args.model, args.threads)
    rows, overview = [], []
    for path in paths:
        started = time.perf_counter()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        result, processed = classifier.classify(image)
        result['decode_and_processing_ms'] = (time.perf_counter() - started) * 1000
        # Labels used only after image-only inference, for reporting.
        expected = path.parent.name if path.parent.name in classifier.names else None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_id = f'{path.stem}_{digest[:8]}'
        input_path = args.output / 'inputs' / f'{artifact_id}.png'
        input_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save(input_path)
        preview = image.copy()
        preview.thumbnail((800, 1000), Image.Resampling.BILINEAR)
        annotated = np.array(preview)[:, :, ::-1].copy()
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 55), (20, 20, 20), -1)
        text = f"type={result['meter_type']} score={max(result['relative_class_scores'].values()):.3f}"
        cv2.putText(annotated, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 240, 100), 2)
        annotated_path = args.output / 'annotated' / f'{artifact_id}.jpg'
        save_image(annotated_path, annotated)
        rows.append({'image': str(path.resolve()), 'source_sha256': digest, 'reference_type': expected,
                     **result, 'exact_match': result['meter_type'] == expected if expected else None,
                     'model_input_image': str(input_path), 'annotated': str(annotated_path)})
        overview.append((annotated_path, f"{path.stem} pred={result['meter_type']} ref={expected or '?'}"))
        print(f'{path.name}: {text} {result["inference_ms"]:.1f}ms', flush=True)
    scored = [r for r in rows if r['exact_match'] is not None]
    summary = {
        'model': str(args.model), 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'model_bytes': args.model.stat().st_size,
        'runtime': f'onnxruntime {ort.__version__}', 'providers': classifier.session.get_providers(),
        'threads': args.threads, 'class_names': classifier.names, 'input_shape': [1, 3, 256, 256],
        'image_count': len(rows), 'labeled_count': len(scored), 'exact_matches': sum(r['exact_match'] for r in scored),
        'by_reference_type': {name: {
            'count': sum(r['reference_type'] == name for r in scored),
            'matched': sum(r['reference_type'] == name and r['exact_match'] for r in scored),
        } for name in classifier.names},
        **{f'median_{key}': float(np.median([r[key] for r in rows]))
           for key in ['inference_ms', 'processing_ms', 'decode_and_processing_ms']},
        'note': 'Final model trained on the current data: this run is a fit check, not held-out accuracy. See results/meter_type_training for disjoint-group fold evaluation. Relative class scores are uncalibrated. Closed-set classifier always selects lcd or wheel; no other-object rejection. No phone validation.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'classifications.json').write_text(json.dumps({'summary': summary, 'images': rows}, indent=2), encoding='utf-8')
    with (args.output / 'classifications.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['image', 'reference_type', 'meter_type', 'exact_match', 'lcd_score', 'wheel_score', 'inference_ms']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in fields}
            record.update(lcd_score=row['relative_class_scores']['lcd'], wheel_score=row['relative_class_scores']['wheel'])
            writer.writerow(record)
    make_overview(overview, args.output / 'overview.jpg', columns=6, tile_w=300, tile_h=380)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
