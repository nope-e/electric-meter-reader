"""Expand original training context and add reviewed screw-only training negatives.

Only uses datas/wheel and original public sources. The old v1 dataset is preserved.
"""
import json
import math
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
import cv2
from main import ROOT,save_image,make_overview
from prepare_wheel_slot_data import sha,build


def main():
    review=json.loads((ROOT/'wheel_slot_reviews.json').read_text())
    for row in review['images']:
        a,b,c,d=row['crop_xyxy']; w,h=c-a,d-b
        row['crop_xyxy']=[math.floor(a-w*.30),math.floor(b-h*.35),math.ceil(c+w*.30),math.ceil(d+h*.35)]
        row['annotation_kind']+=';expanded_real_context_no_new_positive_targets'
    # Pixel coordinates here refer to the inspected 768x1024 orientation-corrected
    # previews of these three original training photographs. Regions contain no wheels.
    definitions={
        '1792416423915':[(62,529,99,570),(439,795,479,836),(220,839,259,881),(202,788,254,834)],
        '1792416424922':[(320,682,357,723),(327,737,362,775)]}
    groups={Path(r['source_image']).stem:r['group'] for r in review['images'] if r['source']=='datas/wheel'}
    negatives=[]; items=[]; patches=[]
    for name,regions in definitions.items():
        path=ROOT.parent/'datas/wheel'/f'{name}.jpg'
        with Image.open(path) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w=rgb.shape[:2]
        for index,region in enumerate(regions):
            a,b,c,d=np.rint(np.asarray(region)*[w/768,h/1024,w/768,h/1024]).astype(int)
            record={'id':f'negative_{name}_{index}','source_image':str(path.resolve()),'source_sha256':sha(path),
                'group':groups[name],'split':'train','count':0,'negative':True,
                'annotation_kind':'assistant_reviewed_screw_or_round_terminal_background_only',
                'crop_xyxy':[int(a),int(b),int(c),int(d)],'boxes':[],'source':'datas/wheel'}
            negatives.append(record)
            patches.append((rgb[b:d,a:c].copy(),record))
            p=ROOT/'.tmp/slot_context_negatives'/f"{record['id']}.jpg"
            save_image(p,rgb[b:d,a:c,::-1]); items.append((p,'background only'))
    # Hard negatives at comparable object scale, beside actual labelled rows.
    # Same deterministic training-only synthesis for 5- and 6-wheel layouts.
    rng=np.random.default_rng(20261020)
    originals=json.loads((ROOT/'datasets/wheel_slots/manifest.json').read_text())
    composites=[]
    generated=ROOT/'datasets/wheel_slot_composites'
    if (ROOT/'wheel_slot_negative_reviews.json').exists(): raise FileExistsError('Completed composite proposal already exists')
    blue_index=0
    for row in originals:
        if row['split']!='train': continue
        if row['source']=='datas/wheel' and row['count']==6:
            blue_index+=1
            if blue_index%4: continue
        with Image.open(ROOT/'datasets/wheel_slots'/row['image']) as im: rgb=np.asarray(im.convert('RGB'))
        h,w=rgb.shape[:2]; dx=round(w*.30); dy=round(h*.35)
        background=np.median(np.concatenate([rgb[:max(1,h//10)].reshape(-1,3),rgb[-max(1,h//10):].reshape(-1,3)]),axis=0)
        canvas=np.empty((h+2*dy,w+2*dx,3),np.uint8); canvas[:]=background
        canvas[dy:dy+h,dx:dx+w]=rgb
        selected=[]
        for side in (0,1):
            patch,record=patches[int(rng.integers(len(patches)))]
            selected.append({'negative_id':record['id'],'source_sha256':record['source_sha256'],'crop_xyxy':record['crop_xyxy'],'group':record['group']})
            ph=max(8,min(round(h*rng.uniform(.40,.62)),round(dx*.8)))
            pw=max(8,min(round(ph*rng.uniform(.82,1.10)),round(dx*.8)))
            patch=cv2.resize(patch,(pw,ph))
            patch=np.clip(patch.astype(float)*rng.uniform(.72,1.2),0,255).astype(np.uint8)
            center_x=round(dx*.48) if side==0 else dx+w+round(dx*.52)
            center_y=dy+round(h*rng.uniform(.4,.6)); x=center_x-pw//2; y=center_y-ph//2
            mask=np.zeros((ph,pw),np.float32)
            cv2.ellipse(mask,(pw//2,ph//2),(max(1,pw//2-1),max(1,ph//2-1)),0,0,360,1.,-1)
            mask=cv2.GaussianBlur(mask,(7,7),1.1)[:,:,None]
            canvas[y:y+ph,x:x+pw]=np.rint(canvas[y:y+ph,x:x+pw]*(1-mask)+patch*mask).astype(np.uint8)
        path=generated/'images'/f"synthetic_{row['id']}.png"; save_image(path,canvas[:,:,::-1])
        boxes=(np.asarray(row['boxes_in_crop'])+[dx,dy,dx,dy]).tolist()
        composites.append({'id':'synthetic_'+row['id'],'source_image':str(path.resolve()),'source_sha256':sha(path),
            'group':row['group'],'split':'train','count':row['count'],'annotation_kind':'training_only_actual_row_with_real_screw_composites',
            'crop_xyxy':[0,0,canvas.shape[1],canvas.shape[0]],'boxes':boxes,'source':'synthetic_training_context',
            'augmentation_sources':{'positive_image':row['source_image'],'positive_source_sha256':row['source_sha256'],'negatives':selected}})
    review['images'].extend(negatives+composites)
    review.update(reviewed=False,review_note='V2: preserve positive annotations, expand actual source context to include logos and unit text. Add 6 manually reviewed screw/terminal-only negatives plus training-only real-row composites. All augmentation sources belong to training groups; no test_wheel image used. Review expanded contexts and synthetic contacts before building.')
    p=ROOT/'wheel_slot_negative_reviews.json'
    if p.exists(): raise FileExistsError(p)
    p.write_text(json.dumps(review,indent=2),encoding='utf-8')
    make_overview(items,ROOT/'.tmp/slot_context_negatives/overview.jpg',columns=4,tile_w=240,tile_h=250)
    items=[(Path(r['source_image']),f"{r['count']} actual wheels + 2 screw backgrounds") for r in composites]
    for start in range(0,len(items),12): make_overview(items[start:start+12],ROOT/'.tmp/slot_context_composites'/f'overview_{start//12:02}.jpg',columns=3,tile_w=600,tile_h=210)


if __name__=='__main__': main()
