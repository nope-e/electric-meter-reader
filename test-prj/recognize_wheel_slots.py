"""Classify -> area -> learned slot boxes -> unchanged VisionState ONNX recognition."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps

from main import ROOT,save_image,make_overview
from detect_wheel_slots import WheelSlotDetector,ordered_slot_crops
from visionstate_wheel import VisionStateWheel


def format_reading(digits,decimal_places=1):
    if decimal_places<0 or decimal_places>=len(digits): raise ValueError('Invalid decimal position')
    return digits[:-decimal_places]+'.'+digits[-decimal_places:] if decimal_places else digits


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--windows',type=Path,help='Replay frozen predicted whole-image classes and area boxes')
    group.add_argument('--input',type=Path,nargs='+',help='Fresh whole-image classification and area detection')
    parser.add_argument('--model',type=Path,default=ROOT/'models/wheel_slots/wheel_slots.onnx')
    parser.add_argument('--output',type=Path,default=ROOT/'results/test_wheel_slot_model_v3')
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--iou',type=float,default=.45)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--area-context',type=float,default=.10,help='Expand each area side by this fraction of its width; vertical padding is 7/6 of this fraction of height')
    parser.add_argument('--slot-margin-x',type=float,default=.08,help='Extra width on each side, as a fraction of the detected slot width')
    parser.add_argument('--slot-margin-y',type=float,default=.10,help='Extra height on each side, as a fraction of the detected slot height')
    parser.add_argument('--decimal-places',type=int,default=1,help='Formatting only; no inferred decimal position')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    if args.decimal_places<0: parser.error('Invalid decimal places')
    if not 0<=args.area_context<=1: parser.error('Invalid area context')
    if not 0<=args.slot_margin_x<=.5 or not 0<=args.slot_margin_y<=.5: parser.error('Invalid slot margin')
    if args.input:
        from detect_meter_area import MeterAreaPipeline
        pipeline=MeterAreaPipeline(threads=args.threads)
        extensions={'.jpg','.jpeg','.png','.webp','.bmp'}
        paths=sorted({p.resolve() for item in args.input for p in (item.rglob('*') if item.is_dir() else [item])
            if p.is_file() and p.suffix.lower() in extensions})
        if not paths: parser.error('No images')
        windows=[]
        for path in paths:
            with Image.open(path) as im: result=pipeline.predict(ImageOps.exif_transpose(im).convert('RGB'))
            windows.append({**result,'image':str(path),'source_sha256':sha(path)})
        args.output.mkdir(parents=True)
        (args.output/'windows.json').write_text(json.dumps({'images':windows},indent=2),encoding='utf-8')
        windows_source={'kind':'fresh_pipeline','models':{'classifier_sha256':sha(ROOT/'models/meter_type_cnn/meter_type.onnx'),
            'area_sha256':sha(ROOT/'models/meter_area/area.onnx')}}
    else:
        args.windows=args.windows or ROOT/'results/test_wheel_area_v2/detections.json'
        windows=json.loads(args.windows.read_text(encoding='utf-8'))['images']
        windows_source={'kind':'frozen_replay','path':str(args.windows),'sha256':sha(args.windows)}
    detector=WheelSlotDetector(args.model,args.threads,args.confidence,args.iou)
    reader=VisionStateWheel(); rows=[]; overviews=[]; strips=[]
    for item in windows:
        path=Path(item['image'])
        if sha(path)!=item['source_sha256']: raise ValueError('Source image changed: '+str(path))
        row={'image':str(path),'source_sha256':item['source_sha256'],'meter_type':item['meter_type'],
            'status':'digital_skipped' if item['meter_type']=='digital' else 'no_area',
            'slots':[],'visionstate_reading_candidate':None,'reasons':[]}
        if item['meter_type']=='wheel' and item['detections']:
            with Image.open(path) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
            h,w=rgb.shape[:2]; box=item['detections'][0]['xyxy']
            pad_x=(box[2]-box[0])*args.area_context
            pad_y=(box[3]-box[1])*args.area_context*7/6
            a,b=max(0,math.floor(box[0]-pad_x)),max(0,math.floor(box[1]-pad_y))
            c,d=min(w,math.ceil(box[2]+pad_x)),min(h,math.ceil(box[3]+pad_y))
            area=rgb[b:d,a:c].copy()
            slots,info=detector.predict(area)
            # Crop from the original photograph, so margin can recover pixels beyond
            # the upstream area boundary. Detection still uses only the area pixels.
            global_boxes=[{'xyxy':(np.asarray(s['xyxy'])+[a,b,a,b]).tolist(),'score':s['score']} for s in slots]
            expanded=ordered_slot_crops(rgb,global_boxes,args.slot_margin_x,args.slot_margin_y)
            for slot,full in zip(slots,expanded):
                slot['pixels']=full['pixels']; slot['crop_xyxy_in_image']=full['crop_xyxy']
                slot['crop_xyxy']=(np.asarray(full['crop_xyxy'])-[a,b,a,b]).tolist()
            row.update(window_xyxy=box,area_crop_xyxy=[a,b,c,d],slot_detection=info,
                       status='review' if slots else 'no_slots')
            artifact=f'{path.stem}_{item["source_sha256"][:8]}'
            preview=area[:,:,::-1].copy()
            if slots:
                prediction,cells=reader.predict([s['pixels'] for s in slots])
                row['prediction']=prediction
                row['raw_digits']=prediction['raw_digits']
                if len(slots)>args.decimal_places:
                    row['visionstate_reading_candidate']=format_reading(prediction['raw_digits'],args.decimal_places)
                else: row['reasons'].append('too_few_slots_for_decimal_format')
                if len(slots) not in (5,6): row['reasons'].append('unusual_slot_count_review_required')
                strip=np.full((180,len(slots)*100,3),25,np.uint8)
                for i,(slot,cell) in enumerate(zip(slots,cells)):
                    crop_path=args.output/'slots'/f'{artifact}_{i}.png'
                    save_image(crop_path,slot['pixels'][:,:,::-1])
                    inp=args.output/'inputs'/f'{artifact}_{i}.png'; save_image(inp,np.rint(cell*255).astype(np.uint8))
                    row['slots'].append({'index':i,'xyxy_in_area':slot['xyxy'],'crop_xyxy_in_area':slot['crop_xyxy'],
                        'crop_xyxy_in_image':slot['crop_xyxy_in_image'],
                        'score':slot['score'],'crop':str(crop_path),'crop_sha256':sha(crop_path),'model_input':str(inp),
                        'model_input_sha256':sha(inp),'independent_state':prediction['independent_states'][i]})
                    x1,y1,x2,y2=np.rint(slot['xyxy']).astype(int)
                    cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),max(1,area.shape[1]//600))
                    cx1,cy1,cx2,cy2=slot['crop_xyxy']
                    cv2.rectangle(preview,(cx1,cy1),(cx2,cy2),(0,180,255),max(1,area.shape[1]//600))
                    cv2.putText(preview,f'{i}:{slot["score"]:.2f}',(x1,max(12,y1)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                    view=cv2.cvtColor(cv2.resize(np.rint(cell*255).astype(np.uint8),(64,96),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                    strip[:96,i*100+18:i*100+82]=view
                    cv2.putText(strip,f'{prediction["independent_states"][i]:.1f}',(i*100+20,120),cv2.FONT_HERSHEY_SIMPLEX,.6,(100,255,100),1)
                cv2.putText(strip,f'joint={prediction["raw_digits"]}',(4,155),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,255,100),1)
                strip_path=args.output/'slot_strips'/f'{artifact}.jpg'; save_image(strip_path,strip)
                strips.append((strip_path,f'{path.stem} {row["visionstate_reading_candidate"]}'))
                row['slot_strip']=str(strip_path)
            preview_path=args.output/'annotated'/f'{artifact}.jpg'; save_image(preview_path,preview)
            row['annotated']=str(preview_path)
            overviews.append((preview_path,f'{path.stem} slots={len(slots)} VS={row["visionstate_reading_candidate"]}'))
        rows.append(row)
        print(path.name,row['status'],len(row['slots']),row['visionstate_reading_candidate'],flush=True)
    completed=[r for r in rows if r.get('prediction')]
    summary={'input_images':len(rows),'predicted_wheel':sum(r['meter_type']=='wheel' for r in rows),
        'with_area':sum('slot_detection' in r for r in rows),'with_predictions':len(completed),
        'slot_count':sum(len(r['slots']) for r in rows),
        'windows_source':windows_source,'slot_model':str(args.model),'slot_model_sha256':detector.sha256,
        'recognizer_sha256':reader.sha256,'runtime':ort.__version__,'threads':args.threads,
        'confidence':args.confidence,'nms_iou':args.iou,'decimal_places':args.decimal_places,
        'area_context_fraction':args.area_context,
        'slot_margin_x':args.slot_margin_x,'slot_margin_y':args.slot_margin_y,
        'median_slot_detector_ms':float(np.median([r['slot_detection']['inference_ms'] for r in rows if 'slot_detection' in r])) if completed else None,
        'note':'Variable slot count from post-NMS model boxes, sorted left to right. No fixed layout, contour splitter, extrapolation, or filename-based rules. Unchanged VisionState model, preprocessing and joint decoding; unfinished last-wheel carry floors. Decimal position is an explicit formatting setting. Detection count and model scores are not accuracy. No phone timing.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'readings.json').write_text(json.dumps({'summary':summary,'images':rows},indent=2),encoding='utf-8')
    with (args.output/'readings.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image','status','slot_count','visionstate_reading_candidate','reasons'])
        writer.writeheader()
        writer.writerows({'image':Path(r['image']).name,'status':r['status'],'slot_count':len(r['slots']),
            'visionstate_reading_candidate':r['visionstate_reading_candidate'],'reasons':';'.join(r['reasons'])} for r in rows)
    for items,name in [(overviews,'overview'),(strips,'slots')]:
        for start in range(0,len(items),12): make_overview(items[start:start+12],args.output/f'{name}_{start//12:02}.jpg',columns=3,tile_w=640,tile_h=240)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
