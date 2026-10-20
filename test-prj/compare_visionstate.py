"""Run VisionState on the exact high-resolution slot crops from a frozen wheel run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from main import ROOT, make_overview, save_image
from visionstate_wheel import VisionStateWheel


def reading(raw):
    return raw[:-1]+'.'+raw[-1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/test_wheel_onnx_v2/readings.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/test_wheel_visionstate')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    reader = VisionStateWheel()
    rows,overviews,strips = [],[],[]
    for item in baseline['images']:
        path = Path(item['image'])
        if sha(path)!=item['source_sha256']:
            raise ValueError(f'Source image changed: {path}')
        row = {k:item.get(k) for k in ('image','source_sha256','status','layout','reasons',
                 'class100_reading_candidate','cont_reading_candidate')}
        row['visionstate_reading_candidate'] = None
        row['slots'] = []
        if item['slots']:
            crops = []
            for slot in item['slots']:
                with Image.open(slot['crop']) as image:
                    crops.append(image.convert('RGB'))
            result,cells = reader.predict(crops)
            row['prediction'] = result
            row['visionstate_reading_candidate'] = reading(result['raw_digits'])
            row['visionstate_independent_candidate'] = reading(result['independent_raw_digits'])
            row['visionstate_author_rounded_candidate'] = reading(result['author_rounded_raw_digits'])
            artifact = f'{path.stem}_{item["source_sha256"][:8]}'
            with Image.open(item['annotated']) as image:
                annotated = np.asarray(image.convert('RGB'))[:,:,::-1].copy()
            # The window and slot geometry are deliberately unchanged from the baseline.
            preview = args.output/'annotated'/f'{artifact}.jpg'
            save_image(preview,annotated)
            label = f"{path.stem} 100={item['class100_raw_digits']} VS={result['raw_digits']}"
            overviews.append((preview,label))
            strip = np.full((180,len(cells)*100,3),25,np.uint8)
            for index,(slot,cell) in enumerate(zip(item['slots'],cells)):
                pixels = np.rint(cell*255).astype(np.uint8)
                input_path = args.output/'inputs'/f'{artifact}_{index}.png'
                save_image(input_path,pixels)
                row['slots'].append({'index':index,'source_crop':slot['crop'],
                     'source_crop_sha256':sha(slot['crop']),'quad_in_area':slot['quad_in_area'],
                     'model_input':str(input_path),'model_input_sha256':sha(input_path)})
                view = cv2.cvtColor(cv2.resize(pixels,(64,96),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                strip[:96,index*100+18:index*100+82] = view
                state = result['independent_states'][index]
                cv2.putText(strip,f'{state:.1f}',(index*100+20,120),cv2.FONT_HERSHEY_SIMPLEX,.6,(100,255,100),1)
            cv2.putText(strip,f"raw={result['independent_raw_digits']} joint={result['raw_digits']}",
                        (4,155),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,255,100),1)
            strip_path = args.output/'slot_strips'/f'{artifact}.jpg'
            save_image(strip_path,strip)
            strips.append((strip_path,label))
            row.update(annotated=str(preview),slot_strip=str(strip_path))
        print(path.name,row['status'],row['class100_reading_candidate'],row['visionstate_reading_candidate'],flush=True)
        rows.append(row)
    completed = [r for r in rows if r.get('prediction')]
    summary = {'input_images':len(rows),'with_predictions':len(completed),
        'slot_count':sum(len(r['slots']) for r in rows),'runtime':'ONNX Runtime CPU',
        'runtime_version':ort.__version__,'threads':1,'model_sha256':reader.sha256,
        'baseline_sha256':sha(args.baseline),
        'model_source':json.loads((ROOT/'models/visionstate/source.json').read_text()),
        'same_reading_as_class100':sum(r['visionstate_reading_candidate']==r['class100_reading_candidate'] for r in completed),
        'median_inference_ms_per_image':float(np.median([r['prediction']['inference_ms'] for r in completed])),
        'note':'Same source images and high-resolution slot crops as frozen baseline. Official preprocessing, slack=1, common shift max=2, penalty=.3. Primary last wheel floors; author rounding saved separately. No new detection, slot tuning, training or previous-reading correction. No full ground truth or accuracy claim.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'readings.json').write_text(json.dumps({'summary':summary,'images':rows},indent=2),encoding='utf-8')
    fields = ['image','status','class100_reading_candidate','cont_reading_candidate',
              'visionstate_reading_candidate','visionstate_independent_candidate','visionstate_author_rounded_candidate']
    with (args.output/'readings.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({**{f:r.get(f) for f in fields},'image':Path(r['image']).name} for r in rows)
    for group,name in ((overviews,'overview'),(strips,'slots')):
        for start in range(0,len(group),12):
            make_overview(group[start:start+12],args.output/f'{name}_{start//12:02}.jpg',columns=3,tile_w=640,tile_h=240)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
