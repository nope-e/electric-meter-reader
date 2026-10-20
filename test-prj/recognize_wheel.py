"""Test pretrained single-wheel classifiers after predicted wheel + area routing."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from main import ROOT, make_overview, save_image


def warp_slot(rgb, center, horizontal, vertical, width, height):
    center, horizontal, vertical = map(lambda a:np.asarray(a,np.float32), (center,horizontal,vertical))
    quad = np.array([center-horizontal*width/2-vertical*height/2,
                     center+horizontal*width/2-vertical*height/2,
                     center+horizontal*width/2+vertical*height/2,
                     center-horizontal*width/2+vertical*height/2],np.float32)
    out_w, out_h = max(2,round(width)),max(2,round(height))
    transform = cv2.getPerspectiveTransform(quad,np.float32([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]]))
    crop = cv2.warpPerspective(rgb,transform,(out_w,out_h),flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return crop, quad.tolist()


def rectangular_window_row(saturation, value, count):
    """Find a strong row of neutral rectangular wheel windows, excluding round screws.

    This is a geometric fallback for the local green layout, not a digit recognizer.
    Multiple saturation masks tolerate glare without relying on a single dark threshold.
    """
    height,width = value.shape
    windows = []
    for limit in (70,100,130):
        mask = np.uint8((saturation<limit)&(value<235))*255
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if not (height*.2<h<height*.95 and width*.025<w<width*.17 and .3<w/h<1.1):
                continue
            rect = cv2.minAreaRect(contour)
            extent = cv2.contourArea(contour)/max(1,rect[1][0]*rect[1][1])
            # A filled circle occupies pi/4 (~.785) of its bounding rectangle.
            if extent<.84:
                continue
            candidate = np.array([x,y,w,h,extent],float)
            overlapping = None
            for index,old in enumerate(windows):
                left,top = np.maximum(candidate[:2],old[:2])
                right,bottom = np.minimum(candidate[:2]+candidate[2:4],old[:2]+old[2:4])
                intersection = max(0,right-left)*max(0,bottom-top)
                union = w*h+old[2]*old[3]-intersection
                if intersection/max(1,union)>.55:
                    overlapping = index
                    break
            if overlapping is None:
                windows.append(candidate)
            elif extent>windows[overlapping][4]:
                windows[overlapping] = candidate
    best = None
    for group in itertools.combinations(sorted(windows,key=lambda b:b[0]),count):
        boxes = np.asarray(group)
        centers = boxes[:,:2]+boxes[:,2:4]/2
        step = (centers[-1]-centers[0])/(count-1)
        pitch = float(np.linalg.norm(step))
        if not width*.08<pitch<width*.23 or abs(step[1])>step[0]*.4:
            continue
        error = float(np.mean(np.linalg.norm(centers-(centers[0]+np.arange(count)[:,None]*step),axis=1))/pitch)
        width_spread = float(np.std(boxes[:,2])/np.mean(boxes[:,2]))
        height_spread = float(np.std(boxes[:,3])/np.mean(boxes[:,3]))
        if error>.1 or width_spread>.15 or height_spread>.2 or np.median(boxes[:,2]/boxes[:,3])>.95:
            continue
        quality = error+width_spread+height_spread+(1-float(np.mean(boxes[:,4])))
        if best is None or quality<best[0]:
            best = (quality,centers,step,pitch,float(np.median(boxes[:,3]))*.8)
    return best


def split_slots(rgb):
    """Pixel-only templates for the two local layouts; no per-file crop/reading data."""
    height,width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,saturation,value = cv2.split(hsv)
    blue = np.count_nonzero((hue>=100)&(hue<=130)&(saturation>65))
    green = np.count_nonzero((hue>=65)&(hue<100)&(saturation>55))
    layout = 'blue_6' if blue>green else 'green_5'
    # Find the white glyph row, avoiding dependence on connected black backgrounds.
    limit = max(100,float(np.percentile(rgb.min(axis=2),65)))
    mask = np.uint8((rgb.min(axis=2)>limit)&(saturation<130))*255
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
                           np.ones((max(1,round(height*.015)),max(1,round(height*.01))),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    red = ((hue<12)|(hue>170))&(saturation>90)&(value>70)
    eligible = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if height*.15<h<height*.9 and width*.015<w<width*.15 and w/h<1.1:
            padding = max(1,w//3)
            patch = red[y:y+h,max(0,x-padding):min(width,x+w+padding)]
            if patch.mean()<.12:
                eligible.append((x,y,w,h))
    count = 5 if layout=='blue_6' else 4
    best = None
    for group in itertools.combinations(sorted(eligible),count):
        boxes = np.array(group,float)
        centers = boxes[:,:2]+boxes[:,2:]/2
        step = (centers[-1]-centers[0])/(count-1)
        pitch = float(np.linalg.norm(step))
        if not width*.08<pitch<width*.23 or abs(step[1])>step[0]*.4:
            continue
        predicted = centers[0]+np.arange(count)[:,None]*step
        error = float(np.mean(np.linalg.norm(predicted-centers,axis=1))/pitch)
        spread = float(np.std(boxes[:,3])/np.mean(boxes[:,3]))
        quality = error+spread*.4-float(np.mean(boxes[:,3])/height)*.08
        if error<.15 and spread<.3 and (best is None or quality<best[0]):
            best = (quality,centers,step,pitch,float(np.median(boxes[:,3])))
    notes = []
    window_row = rectangular_window_row(saturation,value,count) if layout=='green_5' else None
    if window_row is not None:
        _,window_centers,_,window_pitch,_ = window_row
        # Keep existing good crops unchanged. Replace only a missing or displaced glyph row.
        displaced = best is None or np.mean(np.linalg.norm(best[1]-window_centers,axis=1))/window_pitch>.25
        if displaced:
            best = window_row
            notes.append('rectangular_window_row_replaced_missing_or_displaced_glyph_row')
    if best is None:
        return layout,[],['regular_white_glyph_row_not_found']
    _,centers,step,pitch,glyph_height = best
    horizontal = step/pitch
    vertical = np.array([-horizontal[1],horizontal[0]])
    # Fit the wheel pitch from the glyph row; retain margins around complete wheel slots.
    origin = np.mean(centers-np.arange(count)[:,None]*step,axis=0)
    slots = []
    for index in range(count+1):
        center = origin+index*step
        slot_height = glyph_height*1.25
        if index==count and layout=='green_5':
            center = center-vertical*glyph_height*.2
            slot_height = glyph_height*1.8
        slots.append(warp_slot(rgb,center,horizontal,vertical,pitch*.65,slot_height))
    for _,quad in slots:
        points = np.array(quad)
        if np.any(points[:,0]<0) or np.any(points[:,0]>=width) or np.any(points[:,1]<0) or np.any(points[:,1]>=height):
            notes.append('slot_extends_beyond_area')
            break
    return layout,slots,notes


def decode_class100(output):
    values = np.asarray(output).reshape(100)
    state_class = int(np.argmax(values))
    state = state_class/10
    # Chosen convention from the exported dialogue: unfinished carry => preceding digit.
    return {'digit':state_class//10,'state':state,'state_class':state_class,
            'top_class_score':float(values[state_class]),
            'raw_outputs':values.tolist()}


def decode_cont(output):
    values = np.asarray(output).reshape(10)
    index = int(np.argmax(values))
    nxt,prev = (index+1)%10,(index-1)%10
    if values[nxt]>values[prev]:
        fit = float(values[index]+values[nxt])
        state = (index+float(values[nxt])/fit)%10 if fit>0 else None
    else:
        fit = float(values[index]+values[prev])
        state = (index-float(values[prev])/fit)%10 if fit>0 else None
    return {'digit':int(math.floor(state)) if state is not None else None,'state':state,
            'neighbor_fit':fit,'outputs':values.tolist()}


class WheelRecognizer:
    def __init__(self, runtime='onnx'):
        if runtime not in ('onnx','tflite'):
            raise ValueError('Unknown wheel runtime')
        self.runtime = runtime
        self.models = {}
        self.model_records = {}
        for name,stem in [('class100','dig-class100-0182-s2_q'),('cont','dig-cont_0900_s3_q')]:
            path = ROOT/'models/wheel'/f'{stem}.{ "onnx" if runtime=="onnx" else "tflite"}'
            if runtime=='onnx':
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                model = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
                inp,out = model.get_inputs()[0],model.get_outputs()[0]
                if inp.shape!=[1,32,20,3] or inp.type!='tensor(float)':
                    raise ValueError('Unexpected single-wheel input')
                self.models[name] = (model,inp.name,out.name)
                model.run([out.name],{inp.name:np.zeros((1,32,20,3),np.float32)})
                self.runtime_version = ort.__version__
            else:
                from ai_edge_litert.interpreter import Interpreter
                from importlib.metadata import version
                model = Interpreter(model_path=str(path),num_threads=1)
                model.allocate_tensors()
                inp,out = model.get_input_details()[0],model.get_output_details()[0]
                if inp['shape'].tolist()!=[1,32,20,3] or inp['dtype']!=np.float32:
                    raise ValueError('Unexpected single-wheel input')
                model.set_tensor(inp['index'],np.zeros((1,32,20,3),np.float32))
                model.invoke()
                self.models[name] = (model,inp['index'],out['index'])
                self.runtime_version = version('ai-edge-litert')
            self.model_records[name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def predict_slot(self,crop):
        pixels = cv2.resize(crop,(20,32),interpolation=cv2.INTER_LINEAR)
        tensor = pixels.astype(np.float32)[None]
        result = {}
        for name,(model,inp,out) in self.models.items():
            if self.runtime=='onnx':
                start = time.perf_counter()
                output = model.run([out],{inp:tensor})[0][0]
                ms = (time.perf_counter()-start)*1000
            else:
                model.set_tensor(inp,tensor)
                start = time.perf_counter()
                model.invoke()
                ms = (time.perf_counter()-start)*1000
                output = model.get_tensor(out)[0]
            result[name] = (decode_class100(output) if name=='class100' else decode_cont(output))
            result[name]['inference_ms'] = ms
        return result,pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',type=Path,default=ROOT/'results/meter_area_confidence/detections.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/area_wheel_onnx')
    parser.add_argument('--runtime',choices=['onnx','tflite'],default='onnx')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    windows = json.loads(args.windows.read_text(encoding='utf-8'))
    if any(row.get('meter_type') not in ('digital','wheel') for row in windows['images']):
        parser.error('Predicted whole-image types required')
    selected = [row for row in windows['images'] if row['meter_type']=='wheel']
    recognizer = WheelRecognizer(args.runtime)
    results,overview = [],[]
    for item in selected:
        path = Path(item['image'])
        artifact = f'{path.stem}_{item["source_sha256"][:8]}'
        row = {'image':str(path),'meter_type':'wheel','source_sha256':item['source_sha256'],
               'status':'no_area','class100_raw_digits':None,'cont_raw_digits':None,'slots':[]}
        if item['detections']:
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
            box = item['detections'][0]['xyxy']
            x1,y1,x2,y2 = math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])
            area = rgb[y1:y2,x1:x2].copy()
            layout,slots,notes = split_slots(area)
            row.update(window_xyxy=box,layout=layout,reasons=notes,status='review')
            annotated = area[:,:,::-1].copy()
            strip = np.full((160,max(1,len(slots))*120,3),25,np.uint8)
            for index,(crop,quad) in enumerate(slots):
                predicted,pixels = recognizer.predict_slot(crop)
                slot_path = args.output/'slots'/f'{artifact}_{index}.png'
                save_image(slot_path,crop[:,:,::-1])
                input_path = args.output/'inputs'/f'{artifact}_{index}.png'
                save_image(input_path,pixels[:,:,::-1])
                row['slots'].append({'index':index,'quad_in_area':quad,'crop':str(slot_path),
                                     'model_input':str(input_path),**predicted})
                cv2.polylines(annotated,[np.int32(quad)],True,(0,255,0),max(1,area.shape[1]//700))
                view = cv2.resize(pixels[:,:,::-1],(60,96),interpolation=cv2.INTER_NEAREST)
                strip[:96,index*120+30:index*120+90] = view
                cv2.putText(strip,f"{predicted['class100']['state']:.1f}/{predicted['cont']['state']:.1f}",
                            (index*120+4,120),cv2.FONT_HERSHEY_SIMPLEX,.42,(120,255,120),1)
            if slots:
                for kind in ('class100','cont'):
                    raw = ''.join(str(slot[kind]['digit']) if slot[kind]['digit'] is not None else '?' for slot in row['slots'])
                    row[f'{kind}_raw_digits'] = raw
                    row[f'{kind}_reading_candidate'] = raw[:-1]+'.'+raw[-1] if '?' not in raw else None
                row['models_agree'] = row['class100_raw_digits']==row['cont_raw_digits']
                if not row['models_agree']:
                    notes.append('models_disagree')
            else:
                row['status'] = 'no_slots'
            preview = args.output/'annotated'/f'{artifact}.jpg'
            save_image(preview,annotated)
            strip_path = args.output/'slot_strips'/f'{artifact}.jpg'
            save_image(strip_path,strip)
            row.update(annotated=str(preview),slot_strip=str(strip_path))
            overview.append((preview,f"100={row['class100_raw_digits']} cont={row['cont_raw_digits']}"))
        results.append(row)
        print(f"{path.name}: {row['status']} 100={row['class100_raw_digits']} cont={row['cont_raw_digits']}",flush=True)
    completed = [row for row in results if row['slots']]
    summary = {'input_images':len(windows['images']),'predicted_wheel':len(selected),
               'digital_skipped':len(windows['images'])-len(selected),
               'with_area':sum(bool(item['detections']) for item in selected),
               'with_slot_predictions':len(completed),
               'both_models_same_raw_digits':sum(row['models_agree'] for row in completed),
               'windows_sha256':hashlib.sha256(args.windows.read_bytes()).hexdigest(),
               'model_source':json.loads((ROOT/'models/wheel/source.json').read_text()),
               'runtime':recognizer.runtime,'runtime_version':recognizer.runtime_version,
               'models':recognizer.model_records,
               'median_slot_inference_ms':{kind:float(np.median([slot[kind]['inference_ms'] for row in completed for slot in row['slots']])) for kind in ('class100','cont')},
               'note':'All outputs require review. Pixel-based layout templates, no learned slot detector. One trailing decimal assumes the two local wheel layouts. No independent reading references, no accuracy reported.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'readings.json').write_text(json.dumps({'summary':summary,'images':results},indent=2),encoding='utf-8')
    fields = ['image','layout','status','class100_raw_digits','class100_reading_candidate',
              'cont_raw_digits','cont_reading_candidate','models_agree','reasons']
    with (args.output/'readings.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in results:
            record = {field:row.get(field) for field in fields}
            record.update(image=Path(row['image']).name,reasons=';'.join(row.get('reasons',[])))
            writer.writerow(record)
    for start in range(0,len(overview),12):
        make_overview(overview[start:start+12],args.output/f'overview_{start//12:02}.jpg',columns=3,tile_w=640,tile_h=240)
    slot_strips = [(Path(row['slot_strip']),f"100={row['class100_raw_digits']} cont={row['cont_raw_digits']}") for row in completed]
    for start in range(0,len(slot_strips),12):
        make_overview(slot_strips[start:start+12],args.output/f'slots_{start//12:02}.jpg',columns=3,tile_w=640,tile_h=220)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
