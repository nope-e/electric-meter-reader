"""Build a reviewed, single-class wheel-slot dataset from area crops.

Public glyph annotations only propose centers. Explicit reviewed slot indices fill
missing transition/red wheels. Local geometric proposals are reviewed separately.
No images from datas/test_wheel enter this dataset.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageOps
import yaml

from main import ROOT,save_image,make_overview
from merge_window_data import parse_labels,xyxy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_row(centers,indices,count,width,height,last_height=None,last_offset=0):
    indices=np.asarray(indices,float)
    centers=np.asarray(centers,float)
    origin,step=np.linalg.lstsq(np.c_[np.ones(len(indices)),indices],centers,rcond=None)[0]
    centers=origin+np.arange(count)[:,None]*step
    boxes=[]
    for i,(x,y) in enumerate(centers):
        h=last_height if i==count-1 and last_height is not None else height
        if i==count-1: y+=last_offset
        boxes.append([x-width/2,y-h/2,x+width/2,y+h/2])
    return boxes


def proposals():
    manifest=json.loads((ROOT/'datasets/meter_windows/manifest.json').read_text())
    source=ROOT.parent/'datas/Electricity meter.v5i.yolov8'
    names=yaml.safe_load((source/'data.yaml').read_text())['names']
    # Reviewed against every public area crop. Values are spatial slot indices,
    # not digit values. Original labels omit several red/transition wheels.
    public_indices={
        '202304013711200137001E_jpg':(5,[0,1,2,3,4]),
        'img11_jpeg':(6,[0,1,2,3,4,5]),'img13_jpeg':(6,[0,1,2,3,4]),
        'img22_jpeg':(5,[0,1,2,3,4]),'img23_jpeg':(5,[0,1,2,3]),
        'img26_jpeg':(6,[0,1,2,3,4]),'img27_jpeg':(6,[0,1,2,3,4]),
        'img31_jpeg':(6,[0,1,2,3]),'img32_jpeg':(6,[1,2,3,4]),
        'img33_jpeg':(5,[0,1,2,3,4]),'img34_jpeg':(6,[0,1,2,3,4,5]),
        'img36_jpeg':(6,[0,2,3]),'img39_jpeg':(5,[0,1,2,3]),
        'img40_jpeg':(6,[0,1,2,3,4,5]),
        '202304013711200134001E_jpg':(6,[0,1,2,3,4,5]),
        'img24_jpeg':(5,[0,1,2,3]),'img25_jpeg':(6,[0,1,2,3,4]),
        'img30_jpeg':(6,[0,1,2,3,4]),'img35_jpeg':(6,[0,2,5]),
        'img12_jpeg':(6,[1,2,3,4,5]),'img17_jpeg':(5,[0,1,2,3]),
        'img37_jpeg':(5,[0,1,2,3,4]),
    }
    records=[]
    for r in manifest:
        if r['source']!=source.name or not any(w['class']=='wheel' for w in r['windows']): continue
        path=ROOT.parent/'datas'/r['image']
        with Image.open(path) as im: rgb=np.asarray(im.convert('RGB'))
        h,w=rgb.shape[:2]
        labels=parse_labels(path.parent.parent/'labels'/f'{path.stem}.txt',names)
        digits=sorted([l for l in labels if l['class'].isdigit()],key=lambda l:l['box'][0])
        count,indices=public_indices[r['origin']]
        if len(indices)!=len(digits): raise ValueError(r['origin'])
        centers=np.array([d['box'][:2] for d in digits])*[w,h]
        # A complete wheel window, with margin beyond a single currently visible glyph.
        glyph_height=float(np.median([d['box'][3]*h for d in digits]))
        pitch=float(np.linalg.norm((centers[-1]-centers[0])/(indices[-1]-indices[0])))
        boxes=fit_row(centers,indices,count,pitch*.72,glyph_height*1.32)
        win=next(a for a in labels if a['class']=='analog meter')
        a,b,c,d=np.array(xyxy(win['box']))*[w,h,w,h]
        pad=(d-b)*.35
        bounds=[math.floor(a-pad),math.floor(b-pad),math.ceil(c+pad),math.ceil(d+pad)]
        records.append(dict(id='public_'+r['origin'],source_image=str(path.resolve()),
            source_sha256=sha(path),source_label_sha256=r['label_sha256'],
            group='public/'+r['origin'],split=r['split'],count=count,
            annotation_kind='assistant_reviewed_glyph_centers_completed_to_slot_windows',
            crop_xyxy=bounds,boxes=boxes,source='Electricity meter.v5i.yolov8'))
    local=json.loads((ROOT/'results/area_wheel_slots_no_screws/readings.json').read_text())
    groups={Path(r['image']).name:r['group'] for r in json.loads((ROOT/'meter_type_groups.json').read_text())['images']}
    for r in local['images']:
        if not r.get('window_xyxy'): continue
        path=Path(r['image'])
        bounds=[math.floor(r['window_xyxy'][0]),math.floor(r['window_xyxy'][1]),
                math.ceil(r['window_xyxy'][2]),math.ceil(r['window_xyxy'][3])]
        width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
        if r['slots']:
            boxes=[]
            for s in r['slots']:
                q=np.asarray(s['quad_in_area'])
                boxes.append([*q.min(axis=0),*q.max(axis=0)])
            # Need xyxy ordering, not xmin,ymin,xmax,ymax from a rotated crop error.
            boxes=np.asarray(boxes)+[bounds[0],bounds[1],bounds[0],bounds[1]]
            kind='assistant_reviewed_geometry_proposal'
        elif r['layout']=='blue_6':
            # Five blurred blue rows that the white-glyph splitter could not find.
            # Reviewed full rows; manual endpoints are fractions of the area crop.
            endpoints={
                '1792403667969':((.067,.48),(.847,.35),.10,.58),
                '1792416424791':((.083,.52),(.838,.39),.10,.58),
                '1792416424803':((.078,.50),(.843,.42),.10,.59),
                '1792416424819':((.093,.51),(.841,.46),.10,.59),
                '1792416424849':((.083,.48),(.844,.43),.10,.62),
            }
            p0,p1,bw,bh=endpoints[path.stem]
            boxes=np.array(fit_row(np.array([p0,p1])*[width,height],[0,5],6,bw*width,bh*height))
            boxes += [bounds[0],bounds[1],bounds[0],bounds[1]]
            kind='assistant_manual_complete_blurred_row'
        else:
            # This area cuts off the last red wheel; never teach it as background.
            continue
        records.append(dict(id='local_'+path.stem,source_image=str(path.resolve()),
            source_sha256=sha(path),group=groups[path.name],split='train',count=len(boxes),
            annotation_kind=kind,crop_xyxy=bounds,boxes=np.asarray(boxes).tolist(),source='datas/wheel'))
    return records


def build(output,review_path):
    if output.exists(): raise FileExistsError(output)
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if not review.get('reviewed'): raise ValueError('Annotations must be visually reviewed before building')
    rows=[]; items=[]; pixels={}; groups=defaultdict(set)
    for record in review['images']:
        if record.get('exclude'): continue
        path=Path(record['source_image'])
        if ('test_wheel' in path.parts and not record.get('user_authorized_test_wheel_use')) or sha(path)!=record['source_sha256']: raise ValueError(path)
        with Image.open(path) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w=rgb.shape[:2]
        boxes=np.asarray(record['boxes'],float).reshape(-1,4)
        # Expand the crop to contain every manually reviewed wheel window.
        bounds=np.asarray(record['crop_xyxy'],int)
        if len(boxes):
            bounds[:2]=np.minimum(bounds[:2],np.floor(boxes[:,:2].min(axis=0)).astype(int))
            bounds[2:]=np.maximum(bounds[2:],np.ceil(boxes[:,2:].max(axis=0)).astype(int))
        bounds=np.clip(bounds,[0,0,0,0],[w,h,w,h]); a,b,c,d=bounds
        boxes=np.clip(boxes,[a,b,a,b],[c,d,c,d])-[a,b,a,b]
        if len(boxes) not in (0,5,6) or (not len(boxes) and not record.get('negative')) or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]): raise ValueError(record['id'])
        crop=rgb[b:d,a:c].copy(); ch,cw=crop.shape[:2]
        image=output/record['split']/'images'/f"{record['id']}.png"
        label=output/record['split']/'labels'/f"{record['id']}.txt"
        save_image(image,crop[:,:,::-1]); label.parent.mkdir(parents=True,exist_ok=True)
        normalized=np.c_[(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]]/[cw,ch,cw,ch]
        label.write_text(''.join('0 '+' '.join(f'{x:.8f}' for x in row)+'\n' for row in normalized),encoding='utf-8')
        pixel_hash=hashlib.sha256(crop.tobytes()).hexdigest()
        if pixel_hash in pixels: raise ValueError('Duplicate pixels: '+record['id'])
        pixels[pixel_hash]=record['split']; groups[record['group']].add(record['split'])
        row={**record,'crop_xyxy':bounds.tolist(),'image':str(image.relative_to(output)),
             'label':str(label.relative_to(output)),'image_sha256':sha(image),'label_sha256':sha(label),
             'pixel_sha256':pixel_hash,'boxes_in_crop':boxes.tolist(),'size':[cw,ch]}
        rows.append(row)
        preview=crop[:,:,::-1].copy()
        for i,box in enumerate(boxes):
            x1,y1,x2,y2=np.rint(box).astype(int)
            cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,0),max(1,cw//600))
            cv2.putText(preview,str(i),(x1,max(12,y1)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
        p=output/'previews'/f"{record['id']}.jpg"; save_image(p,preview)
        items.append((p,f"{record['split']} slots={len(boxes)}"))
    if any(len(s)>1 for s in groups.values()): raise ValueError('Source group split leakage')
    for start in range(0,len(items),12): make_overview(items[start:start+12],output/f'preview_{start//12:02}.jpg',columns=3,tile_w=600,tile_h=210)
    (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output.resolve()),'train':'train/images',
        'val':'valid/images','test':'test/images','names':{0:'wheel_slot'}},sort_keys=False),encoding='utf-8')
    summary={'images':len(rows),'slots':sum(r['count'] for r in rows),
        'splits':{s:{'images':sum(r['split']==s for r in rows),'slots':sum(r['count'] for r in rows if r['split']==s)} for s in ['train','valid','test']},
        'counts':dict(Counter(r['count'] for r in rows)), 'review_sha256':sha(review_path),
        'annotation_note':'Assistant visually reviewed geometry proposals and completed missing public wheels; not independently human-audited. '+('User authorized selected test_wheel groups for adaptation; see adaptation_groups. This development set is no longer untouched.' if review.get('adaptation_groups') else 'Test_wheel never trained.')+' Public original splits retained; both repeated original local meter groups entirely train. Public crops come from reviewed parent areas; local crops from frozen predicted area. Explicit negative records, if present, contain background only.',
        'adaptation_groups':review.get('adaptation_groups',{}),
        'source_metadata':yaml.safe_load((ROOT.parent/'datas/Electricity meter.v5i.yolov8/data.yaml').read_text())['roboflow']}
    (output/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/wheel_slots')
    p.add_argument('--review',type=Path,default=ROOT/'wheel_slot_reviews.json')
    p.add_argument('--propose',action='store_true')
    args=p.parse_args()
    if args.propose:
        if args.review.exists(): raise FileExistsError(args.review)
        args.review.write_text(json.dumps({'reviewed':False,'images':proposals()},indent=2),encoding='utf-8')
    else: build(args.output,args.review)


if __name__=='__main__': main()
