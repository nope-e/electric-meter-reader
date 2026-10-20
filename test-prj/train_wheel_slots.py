"""Train YOLO11n wheel-slot detection, evaluate holdout, export and verify ONNX."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
for key,relative in {'YOLO_CONFIG_DIR':'.cache/ultralytics','TORCH_HOME':'.cache/torch',
                     'TEMP':'.tmp','TMP':'.tmp','MPLCONFIGDIR':'.cache/matplotlib'}.items():
    p=ROOT/relative; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
os.environ['YOLO_AUTOINSTALL']='false'

import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO,settings
from main import preprocess,read_image,decode
from train_meter_windows import write_json,metrics_record


def verify_data(data):
    manifest=json.loads((data/'manifest.json').read_text())
    groups={}; hashes={}
    training_sources={r['source_sha256'] for r in manifest if r['split']=='train' and r['source']!='synthetic_training_context'}
    for row in manifest:
        for key in ['image','label']:
            if hashlib.sha256((data/row[key]).read_bytes()).hexdigest()!=row[key+'_sha256']:
                raise ValueError('Changed data: '+row[key])
        if 'test_wheel' in Path(row['source_image']).parts and not row.get('user_authorized_test_wheel_use'):
            raise ValueError('Test data use must be explicitly authorized and grouped')
        if hashlib.sha256(Path(row['source_image']).read_bytes()).hexdigest()!=row['source_sha256']:
            raise ValueError('Changed source image: '+row['source_image'])
        lines=[line for line in (data/row['label']).read_text().splitlines() if line.strip()]
        if len(lines)!=row['count'] or (not lines and not row.get('negative')):
            raise ValueError('Unexpected positive/negative labels')
        augmentation=row.get('augmentation_sources')
        if augmentation:
            if row['split']!='train' or augmentation['positive_source_sha256'] not in training_sources:
                raise ValueError('Composite positive is not training-only')
            if any(n['source_sha256'] not in training_sources for n in augmentation['negatives']):
                raise ValueError('Composite background is not training-only')
        for value,mapping in [(row['group'],groups),(row['pixel_sha256'],hashes)]:
            if value in mapping and mapping[value]!=row['split']: raise ValueError('Split leakage')
            mapping[value]=row['split']
    return manifest


def verify_export(weight,exported,manifest,data):
    onnx.checker.check_model(str(exported))
    torch.set_num_threads(4)
    model=YOLO(str(weight)); model.model.cpu().float().eval()
    options=ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0]
    if inp.shape!=[1,3,640,640] or model.names!={0:'wheel_slot'}: raise ValueError('Bad model contract')
    chosen=[]
    for split in ['train','valid','test']:
        chosen.extend([r for r in manifest if r['split']==split][:3])
    checks=[]
    for row in chosen:
        image=read_image(data/row['image']); x,scale,left,top=preprocess(image,640,640)
        with torch.inference_mode(): raw=model.model(torch.from_numpy(x))[0].numpy()
        actual=session.run(None,{inp.name:x})[0]
        np.testing.assert_allclose(actual,raw,rtol=1e-3,atol=2e-3)
        pred,_=decode(actual,scale,left,top,image.shape,{0:'wheel_slot'},.25,.45)
        ref=model.predict(image,imgsz=640,rect=False,device='cpu',conf=.25,iou=.45,verbose=False)[0]
        # Sort by position: insignificant score changes can reorder equal-confidence boxes.
        a=sorted(pred,key=lambda d:d['xyxy'][0]); b=sorted(ref.boxes.xyxy.numpy().tolist(),key=lambda b:b[0])
        if len(a)!=len(b): raise ValueError('PT/ONNX count mismatch')
        if a: np.testing.assert_allclose([d['xyxy'] for d in a],b,atol=.1,rtol=1e-3)
        checks.append({'image':row['image'],'max_abs_error':float(np.abs(raw-actual).max()),'detections':len(a),'passed':True})
    return {'input':inp.shape,'output':session.get_outputs()[0].shape,'checks':checks,'runtime':ort.__version__}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'datasets/wheel_slots')
    p.add_argument('--weights',type=Path,default=ROOT/'models/meter_area/area.pt')
    p.add_argument('--name',default='slots_yolo11n_v1')
    p.add_argument('--epochs',type=int,default=160)
    p.add_argument('--patience',type=int,default=35)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--device',default='0')
    args=p.parse_args(); manifest=verify_data(args.data)
    project=ROOT/'models/wheel_slots'; project.mkdir(parents=True,exist_ok=True)
    run=project/args.name
    if run.exists(): raise FileExistsError(run)
    if args.device!='cpu' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    settings.update({k:False for k in ['sync','wandb','mlflow','comet','clearml','dvc','hub','tensorboard','neptune','vscode_msg','openvino_msg'] if k in settings})
    status=project/f'{args.name}.status.json'; started=time.perf_counter()
    state={'phase':'starting','base_weights':str(args.weights),'base_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
           'dataset_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
           'epochs_requested':args.epochs,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if args.device!='cpu' else None}
    def progress(trainer):
        with trainer.csv.open(newline='') as f: completed=list(csv.DictReader(f))
        state.update(phase='training',epoch=int(float(completed[-1]['epoch'])),
            elapsed_seconds=time.perf_counter()-started,metrics={k:float(v) for k,v in trainer.metrics.items()})
        write_json(status,state)
        print('EPOCH_STATUS '+json.dumps(state),flush=True)
    try:
        model=YOLO(str(args.weights)); model.add_callback('on_fit_epoch_end',progress)
        request=dict(data=str((args.data/'data.yaml').resolve()),epochs=args.epochs,patience=args.patience,
            imgsz=640,batch=args.batch,device=args.device,workers=0,project=str(project),name=args.name,
            optimizer='AdamW',lr0=.001,lrf=.01,cos_lr=True,seed=20261020,deterministic=True,amp=False,
            degrees=8.,translate=.08,scale=.25,shear=2.,perspective=.0002,fliplr=0.,flipud=0.,
            hsv_h=.015,hsv_s=.35,hsv_v=.35,mosaic=.3,close_mosaic=15,mixup=0.,
            cache=False,plots=True,save=True)
        write_json(project/f'{args.name}.request.json',request); write_json(status,state)
        model.train(**request)
        best=run/'weights/best.pt'
        state.update(phase='evaluating',elapsed_seconds=time.perf_counter()-started); write_json(status,state)
        metrics=YOLO(str(best)).val(data=request['data'],split='test',imgsz=640,batch=args.batch,
            device=args.device,workers=0,project=str(run),name='test',plots=True)
        write_json(run/'test_metrics.json',metrics_record(metrics))
        exported=Path(YOLO(str(best)).export(format='onnx',imgsz=640,batch=1,dynamic=False,
            simplify=False,opset=12,nms=False,device='cpu',half=False))
        verification=verify_export(best,exported,manifest,args.data)
        write_json(run/'onnx_verification.json',verification)
        shutil.copy2(exported,project/'wheel_slots.onnx'); shutil.copy2(best,project/'wheel_slots.pt')
        with (run/'results.csv').open(newline='') as f: epochs=list(csv.DictReader(f))
        state.update(phase='complete',epochs_completed=len(epochs),best_checkpoint=str(best),
            best_epoch=int(float(max(epochs,key=lambda r:float(r['metrics/mAP50-95(B)']))['epoch'])),
            elapsed_seconds=time.perf_counter()-started,test_metrics=metrics_record(metrics),
            onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),onnx_bytes=exported.stat().st_size)
        write_json(project/'source.json',{'task':'complete wheel-slot detection in an area crop',
            'classes':{0:'wheel_slot'},'architecture':'YOLO11n','training':state,
            'dataset_summary':json.loads((args.data/'summary.json').read_text()),
            'request':request,'input':'float32 RGB NCHW [1,3,640,640], /255, letterbox fill 114',
            'output':'[1,5,8400], xywh and wheel_slot score, no embedded NMS',
            'selection':'confidence .25, NMS IoU .45, all boxes ordered by center x; variable count; no extrapolation',
            'upstream_base_source':'models/display.source.json; existing meter_area fine-tune',
            'recognizer':'VisionState unchanged, separate model; see models/visionstate/source.json'})
        write_json(status,state); print(json.dumps(state,indent=2),flush=True)
    except Exception as exc:
        state.update(phase='failed',error=repr(exc),elapsed_seconds=time.perf_counter()-started)
        write_json(status,state); raise


if __name__=='__main__': main()
