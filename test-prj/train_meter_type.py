"""Fine-tune a small LCD/wheel classifier; evaluate with two fixed disjoint folds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
for key, relative in {"TORCH_HOME": ".cache/torch", "TEMP": ".tmp", "TMP": ".tmp"}.items():
    (ROOT / relative).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(ROOT / relative)

import numpy as np
from PIL import Image, ImageOps
import torch
import onnx
import onnxruntime as ort
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NAMES = ["lcd", "wheel"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def meter_tensor(image):
    # Keep the whole frame rather than removing the counter with a center crop.
    fitted = ImageOps.pad(image, (256, 256), method=Image.Resampling.BILINEAR, color=(128, 128, 128))
    return transforms.functional.normalize(transforms.functional.to_tensor(fitted), MEAN, STD)


class MeterDataset(Dataset):
    def __init__(self, images, rows, indices, augment):
        self.images, self.rows, self.indices = images, rows, list(indices)
        self.augment = transforms.Compose([
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.90, 1.10), fill=128),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
            transforms.RandomGrayscale(p=0.2),
        ]) if augment else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        image = self.images[i].copy()
        if self.augment:
            image = self.augment(image)
        tensor = transforms.functional.normalize(transforms.functional.to_tensor(image), MEAN, STD)
        return tensor, NAMES.index(self.rows[i]["reference_type"])


def make_model(state):
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(state)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model


def train(images, rows, indices, state, epochs, name):
    torch.manual_seed(20261019)
    random.seed(20261019)
    np.random.seed(20261019)
    model = make_model(state)
    dataset = MeterDataset(images, rows, indices, augment=True)
    labels = [NAMES.index(rows[i]["reference_type"]) for i in indices]
    counts = np.bincount(labels, minlength=2)
    weights = [1 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=64, replacement=True,
                                    generator=torch.Generator().manual_seed(20261019))
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        # Tiny training sets cannot estimate reliable batch-normalization statistics.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        losses = []
        for tensor, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(tensor), label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"{name}: epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return model.eval()


def export_and_check(model, path, images, rows, indices):
    torch.onnx.export(model, torch.zeros(1, 3, 256, 256), str(path),
                      input_names=['images'], output_names=['logits'],
                      opset_version=12, dynamo=False)
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {
        'class_names': json.dumps(NAMES),
        'preprocess_kind': 'rgb_imagenet_pad256',
        'preprocessing': json.dumps({'size': [256, 256], 'resize': 'PIL bilinear aspect-preserving pad',
                                     'pad_rgb': [128, 128, 128], 'mean': MEAN, 'std': STD,
                                     'layout': 'NCHW', 'dtype': 'float32', 'pixel_divisor': 255}),
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    result, largest_error = [], 0.0
    with torch.inference_mode():
        for i in indices:
            tensor = meter_tensor(images[i]).unsqueeze(0)
            expected = model(tensor).numpy()
            actual = session.run(None, {'images': tensor.numpy()})[0]
            largest_error = max(largest_error, float(np.max(np.abs(expected - actual))))
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
            shifted = actual[0] - actual[0].max()
            scores = np.exp(shifted) / np.exp(shifted).sum()
            predicted = NAMES[int(scores.argmax())]
            assert predicted == NAMES[int(expected[0].argmax())]
            result.append({**rows[i], 'predicted_type': predicted,
                           'exact_match': predicted == rows[i]['reference_type'],
                           'relative_scores': dict(zip(NAMES, map(float, scores)))})
    print(f'{path.name}: ONNX checked on {len(result)} inputs, max_abs_error={largest_error:.8f}', flush=True)
    return result, {'max_abs_logit_error': largest_error, 'count': len(result), 'rtol': 1e-4, 'atol': 2e-5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "datas")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--groups", type=Path, default=ROOT / 'meter_type_groups.json')
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    state = MobileNet_V3_Small_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    if args.download_only:
        return
    directory = ROOT / "models/meter_type_cnn"
    output = ROOT / "results/meter_type_training"
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for label in NAMES for p in sorted((args.input / label).glob("*.jpg"))]
    rows = [{"image": str(p.resolve()), "reference_type": p.parent.name,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    manifest = json.loads(args.groups.read_text(encoding='utf-8'))
    groups = {r['image']: r for r in manifest['images']}
    assert len(groups) == len(manifest['images']), 'Duplicate manifest entries'
    actual_names = {p.relative_to(args.input).as_posix() for p in paths}
    assert actual_names == set(groups), 'Update group manifest to match current photos before training'
    for path, row in zip(paths, rows):
        group = groups[path.relative_to(args.input).as_posix()]
        row.update(group=group['group'], fold=group['fold'])
    splits = np.array([r['fold'] for r in rows])
    assert set(splits) == {0, 1}
    for fold in range(2):
        train_rows, test_rows = [r for r in rows if r['fold'] != fold], [r for r in rows if r['fold'] == fold]
        assert not {r['group'] for r in train_rows} & {r['group'] for r in test_rows}, 'Group leakage'
        assert not {r['sha256'] for r in train_rows} & {r['sha256'] for r in test_rows}, 'Exact duplicate leakage'
        assert {r['reference_type'] for r in train_rows} == set(NAMES)
        assert {r['reference_type'] for r in test_rows} == set(NAMES)
        print(f'fold{fold}: train={len(train_rows)} held-out={len(test_rows)} groups={sorted({r["group"] for r in test_rows})}', flush=True)
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            images.append(ImageOps.pad(image, (256, 256), method=Image.Resampling.BILINEAR, color=(128, 128, 128)))
    predictions = []
    verification = {}
    for fold in range(2):
        train_indices = np.flatnonzero(splits != fold)
        test_indices = np.flatnonzero(splits == fold)
        model = train(images, rows, train_indices, state, args.epochs, f"fold{fold}")
        torch.save(model.state_dict(), directory / f"fold{fold}.pt")
        result, verification[f'fold{fold}'] = export_and_check(model, directory / f'fold{fold}.onnx', images, rows, test_indices)
        for row in result:
            row["held_out_fold"] = fold
            row["train_images"] = [rows[i]["image"] for i in train_indices]
        predictions.extend(result)
        print(f"fold{fold}: held-out matches={sum(r['exact_match'] for r in result)}/{len(result)}", flush=True)
        (output / "held_out_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    model = train(images, rows, list(range(len(rows))), state, args.epochs, "all_data")
    torch.save(model.state_dict(), directory / "meter_type.pt")
    fitted, verification['all_data'] = export_and_check(model, directory / 'meter_type.onnx', images, rows, range(len(rows)))
    (output / 'fit_predictions.json').write_text(json.dumps(fitted, indent=2), encoding='utf-8')
    (output / 'export_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    metadata = {"architecture": "torchvision mobilenet_v3_small", "class_names": NAMES,
                "pretrained_url": MobileNet_V3_Small_Weights.DEFAULT.url, "epochs": args.epochs,
                "seed": 20261019, "threads": args.threads,
                "input": "Full image aspect-preserving pad to 256x256 gray128, RGB /255, ImageNet mean/std.",
                "cached_training_images": "Full original RGB images directly aspect-preserving padded to 256x256 before training augmentation, same unaugmented preprocessing as inference",
                "training_images": rows, "fold_assignment": splits.tolist(),
                "held_out_matches": sum(r["exact_match"] for r in predictions), "held_out_count": len(predictions),
                "held_out_by_type": {label: {"count": sum(r["reference_type"] == label for r in predictions),
                                               "matched": sum(r["reference_type"] == label and r["exact_match"] for r in predictions)} for label in NAMES},
                "group_manifest": manifest,
                "export_verification": verification,
                "fit_matches": sum(r['exact_match'] for r in fitted),
                "note": "Exploratory two-fold group validation; only two wheel appearance families, one per training fold. Epoch count fixed before evaluation; no validation-based checkpoint selection. Held-out scores use fold ONNX models; final model is fit on all images. No independent dataset or phone validation."}
    (directory / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ["held_out_matches", "held_out_count", "held_out_by_type"]}, indent=2))


if __name__ == "__main__":
    main()
