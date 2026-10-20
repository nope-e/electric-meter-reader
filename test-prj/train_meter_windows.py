"""Train typed windows or a single area class, test the best checkpoint and export ONNX."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
for key, relative in {'YOLO_CONFIG_DIR':'.cache/ultralytics', 'TORCH_HOME':'.cache/torch',
                      'TEMP':'.tmp', 'TMP':'.tmp', 'MPLCONFIGDIR':'.cache/matplotlib'}.items():
    directory = ROOT/relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
os.environ['YOLO_AUTOINSTALL'] = 'false'

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO, settings

from main import decode, preprocess, read_image
from merge_window_data import verify_dataset


def write_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def metrics_record(metrics):
    box = metrics.box
    return {'aggregate':{k:float(v) for k,v in metrics.results_dict.items()},
            'per_class':[{'class_id':int(i), 'class_name':metrics.names[int(i)],
                          'precision':float(box.p[j]), 'recall':float(box.r[j]),
                          'map50':float(box.ap50[j]), 'map50_95':float(box.ap[j].mean())}
                         for j,i in enumerate(box.ap_class_index)]}


def verify_onnx(weight, exported, data):
    torch.set_num_threads(4)
    model = YOLO(str(weight))
    model.model.cpu().eval().float()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(exported), sess_options=options, providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    if inp.shape != [1,3,640,640]:
        raise ValueError(f'Unexpected ONNX input: {inp.shape}')
    names = yaml.safe_load(data.read_text(encoding='utf-8'))['names']
    expected_names = dict(enumerate(names)) if isinstance(names, list) else names
    if model.names != expected_names:
        raise ValueError(f'Unexpected classes: {model.names}')
    manifest = json.loads((data.parent/'manifest.json').read_text(encoding='utf-8'))
    chosen = []
    for name in ['lcd','wheel']:
        candidates = [r for r in manifest if r['split'] == 'test'
                      and any(b.get('source_meter_type', b['class']) == name for b in r['windows'])]
        chosen.extend(candidates[:3])
    checks = []
    for row in chosen:
        path = data.parent/row['output_image']
        image = read_image(path)
        tensor, scale, left, top = preprocess(image, 640, 640)
        with torch.inference_mode():
            raw = model.model(torch.from_numpy(tensor))
            raw = (raw[0] if isinstance(raw,(tuple,list)) else raw).numpy()
        actual = session.run(None, {inp.name:tensor})[0]
        np.testing.assert_allclose(actual, raw, rtol=1e-3, atol=2e-3)
        detections, _ = decode(actual, scale, left, top, image.shape, model.names, .25, .45)
        reference = model.predict(image, imgsz=640, rect=False, conf=.25, iou=.45,
                                  device='cpu', verbose=False)[0]
        if len(detections) != len(reference.boxes):
            raise ValueError(f'PT/ONNX detection count mismatch: {path}')
        if detections:
            np.testing.assert_allclose([d['xyxy'] for d in detections],
                                       reference.boxes.xyxy.numpy(), rtol=1e-3, atol=.1)
            if [d['class_id'] for d in detections] != reference.boxes.cls.int().tolist():
                raise ValueError(f'PT/ONNX class mismatch: {path}')
        checks.append({'image':path.name, 'max_abs_error':float(np.abs(actual-raw).max()),
                       'detections':len(detections), 'passed':True})
    onnx.checker.check_model(str(exported))
    return {'input':inp.shape, 'output':session.get_outputs()[0].shape,
            'checks':checks, 'runtime':ort.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT/'models/display.pt')
    parser.add_argument('--name')
    parser.add_argument('--single-area', action='store_true', help='Single area class; whole-image classification is separate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    dataset_name = 'meter_area' if args.single_area else 'meter_windows'
    artifact_name = 'area' if args.single_area else 'meter_windows'
    args.data = (args.data or ROOT/f'datasets/{dataset_name}/data.yaml').resolve()
    args.name = args.name or ('area_yolo11n' if args.single_area else 'reading_yolo11n')
    project = ROOT/f'models/{dataset_name}'
    project.mkdir(parents=True, exist_ok=True)
    run = project/args.name
    status_path = project/f'{args.name}.status.json'
    if run.exists() and not args.resume:
        raise FileExistsError(f'Existing run: {run}; use --resume or a new --name')
    verify_dataset(args.data.parent, ROOT.parent/'datas')
    torch.set_num_threads(4)
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use GPU PyTorch or explicitly --device cpu')
    settings.update({k:False for k in ['sync','wandb','mlflow','comet','clearml','dvc',
                                      'hub','tensorboard','neptune','vscode_msg','openvino_msg']
                     if k in settings})
    started = time.perf_counter()
    state = {'phase':'starting', 'run':str(run), 'epochs_requested':args.epochs,
             'torch':torch.__version__, 'device':args.device,
             'gpu':torch.cuda.get_device_name(0) if args.device != 'cpu' else None,
             'data':str(args.data), 'base_weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest()}
    write_json(status_path, state)

    def progress(trainer):
        # The final evaluation callback can run after trainer.epoch advances.
        # Use the recorded completed epoch rather than counting that callback.
        with trainer.csv.open(encoding='utf-8', newline='') as stream:
            completed = list(csv.DictReader(stream))
        state.update(phase='training', epoch=int(float(completed[-1]['epoch'])),
                     elapsed_seconds=time.perf_counter()-started,
                     best_fitness=float(trainer.best_fitness),
                     metrics={k:float(v) for k,v in trainer.metrics.items()})
        write_json(status_path, state)
        print('EPOCH_STATUS '+json.dumps(state), flush=True)

    def stage(phase):
        state.update(phase=phase, elapsed_seconds=time.perf_counter()-started)
        write_json(status_path, state)

    try:
        model = YOLO(str(run/'weights/last.pt' if args.resume else args.weights))
        model.add_callback('on_fit_epoch_end', progress)
        kwargs = dict(data=str(args.data), epochs=args.epochs, patience=args.patience,
                      imgsz=640, batch=args.batch, device=args.device, workers=args.workers,
                      project=str(project), name=args.name, exist_ok=False,
                      optimizer='AdamW', lr0=.001, lrf=.01, cos_lr=True,
                      seed=20261019, deterministic=True, amp=False,
                      degrees=10., translate=.1, scale=.25, fliplr=0., flipud=0.,
                      hsv_s=.3, hsv_v=.3, mosaic=.5, close_mosaic=10,
                      mixup=0., cache=False, plots=True, save=True)
        write_json(project/f'{args.name}.request.json', kwargs)
        model.train(**({'resume':True, 'device':args.device} if args.resume else kwargs))
        with (run/'results.csv').open(encoding='utf-8', newline='') as stream:
            completed = list(csv.DictReader(stream))
        state.update(epochs_completed=len(completed), epoch=int(float(completed[-1]['epoch'])),
                     best_epoch=int(float(max(completed, key=lambda r: float(r['metrics/mAP50-95(B)']))['epoch'])))
        stage('testing_best_checkpoint')
        best = run/'weights/best.pt'
        selected = YOLO(str(best))
        test = selected.val(data=str(args.data), split='test', imgsz=640, batch=args.batch,
                            device=args.device, workers=args.workers, plots=True,
                            project=str(run), name='test', exist_ok=False)
        write_json(run/'test_metrics.json', metrics_record(test))
        stage('exporting_onnx')
        exported = Path(YOLO(str(best)).export(format='onnx', imgsz=640, batch=1,
                         dynamic=False, simplify=False, opset=12, nms=False,
                         device='cpu', half=False))
        stage('verifying_onnx')
        verification = verify_onnx(best, exported, args.data)
        write_json(run/'onnx_verification.json', verification)
        shutil.copy2(exported, project/f'{artifact_name}.onnx')
        shutil.copy2(best, project/f'{artifact_name}.pt')
        state.update(phase='complete', elapsed_seconds=time.perf_counter()-started,
                     best_checkpoint=str(best), onnx=str(project/f'{artifact_name}.onnx'),
                     test_metrics=metrics_record(test), onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                     onnx_bytes=exported.stat().st_size)
        write_json(status_path, state)
        print(json.dumps(state, indent=2), flush=True)
    except Exception as exc:
        state.update(phase='failed', error=repr(exc), elapsed_seconds=time.perf_counter()-started)
        write_json(status_path, state)
        raise


if __name__ == '__main__':
    main()
