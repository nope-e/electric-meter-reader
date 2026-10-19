"""Verify actual image preprocessing, PyTorch/ORT parity and held-out predictions."""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from classify_meter import MeterTypeClassifier, preprocess_meter
from main import ROOT, make_overview, save_image
from train_meter_type import meter_tensor


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = ROOT / 'models/meter_type_cnn'
    output = ROOT / 'results/meter_type_training'
    training = json.loads((models / 'training.json').read_text(encoding='utf-8'))
    expected = {r['image']: r for r in json.loads((output / 'held_out_predictions.json').read_text(encoding='utf-8'))}
    classifiers, sources = {}, {}
    for name in ['fold0', 'fold1', 'meter_type']:
        classifiers[name] = MeterTypeClassifier(models / f'{name}.onnx')
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model.load_state_dict(torch.load(models / f'{name}.pt', map_location='cpu', weights_only=True))
        sources[name] = model.eval()
    rows, overview, errors = [], [], []
    max_preprocess, max_logits, max_scores = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for row in training['training_images']:
            path = Path(row['image'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], f'Source changed: {path}'
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            tensor, _ = preprocess_meter(image)
            reference_tensor = meter_tensor(image).unsqueeze(0)
            np.testing.assert_allclose(tensor, reference_tensor.numpy(), rtol=0, atol=1e-6)
            max_preprocess = max(max_preprocess, float(np.abs(tensor - reference_tensor.numpy()).max()))
            fold_name = f'fold{row["fold"]}'
            for name in [fold_name, 'meter_type']:
                classifier = classifiers[name]
                reference_logits = sources[name](reference_tensor).numpy()
                logits = classifier.session.run(None, {classifier.input_name: tensor})[0]
                np.testing.assert_allclose(logits, reference_logits, rtol=1e-4, atol=2e-5)
                max_logits = max(max_logits, float(np.abs(logits - reference_logits).max()))
            held_out, _ = classifiers[fold_name].classify(image)
            prior = expected[row['image']]
            assert held_out['meter_type'] == prior['predicted_type']
            for label, score in held_out['relative_class_scores'].items():
                difference = abs(score - prior['relative_scores'][label])
                max_scores = max(max_scores, difference)
                assert difference < 1e-5
            final, _ = classifiers['meter_type'].classify(image)
            matched = held_out['meter_type'] == row['reference_type']
            record = {**row, **held_out, 'held_out_fold': row['fold'], 'exact_match': matched,
                      'final_fit_type': final['meter_type'], 'final_fit_match': final['meter_type'] == row['reference_type']}
            rows.append(record)
            image.thumbnail((800, 1000), Image.Resampling.BILINEAR)
            annotated = np.array(image)[:, :, ::-1].copy()
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 55), (20, 20, 20), -1)
            label = f"held-out={held_out['meter_type']} ref={row['reference_type']} fold={row['fold']}"
            cv2.putText(annotated, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (80, 240, 80) if matched else (80, 80, 255), 2)
            target = output / 'held_out_annotated' / path.name
            save_image(target, annotated)
            item = (target, f"{'OK' if matched else 'ERROR'} {held_out['meter_type']} ref={row['reference_type']}")
            overview.append(item)
            if not matched:
                errors.append(item)
    make_overview(overview, output / 'held_out_overview.jpg', columns=6, tile_w=300, tile_h=380)
    if errors:
        make_overview(errors, output / 'held_out_errors.jpg', columns=4, tile_w=420, tile_h=500)
    summary = {
        'held_out_matches': sum(r['exact_match'] for r in rows), 'count': len(rows),
        'held_out_by_type': {label: {'count': sum(r['reference_type'] == label for r in rows),
                                    'matched': sum(r['reference_type'] == label and r['exact_match'] for r in rows)}
                             for label in ['lcd', 'wheel']},
        'held_out_by_wheel_group': {group: {'count': sum(r['group'] == group for r in rows),
                                           'matched': sum(r['group'] == group and r['exact_match'] for r in rows)}
                                    for group in sorted({r['group'] for r in rows if r['reference_type'] == 'wheel'})},
        'final_fit_matches': sum(r['final_fit_match'] for r in rows),
        'max_preprocess_abs_error': max_preprocess,
        'max_pytorch_ort_abs_logit_error': max_logits, 'max_repeat_score_difference': max_scores,
        'torch_version': torch.__version__,
        'model_sha256': {name: hashlib.sha256((models / f'{name}.onnx').read_bytes()).hexdigest() for name in classifiers},
        'note': 'Held-out scores from disjoint-group fold models; final-fit scores from all-data model are resubstitution, not validation. Only two wheel appearance groups. No phone testing.',
    }
    (output / 'runtime_verification.json').write_text(json.dumps({'summary': summary, 'images': rows}, indent=2), encoding='utf-8')
    with (output / 'held_out_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        keys = ['image', 'reference_type', 'group', 'held_out_fold', 'meter_type', 'exact_match', 'final_fit_type', 'final_fit_match']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
