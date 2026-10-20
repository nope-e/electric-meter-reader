"""User-authorized grouped adaptation using part of test_wheel, with full-window boxes."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageOps
from main import ROOT,save_image,make_overview
from prepare_wheel_slot_data import sha


GROUPS={
    'new_a':('train',['1792476679494','1792476679554','1792476679632']),
    'new_b':('train',['1792476679923','1792476679989']),
    'new_c':('train',['1792476680036','1792476680134']),
    'new_d':('valid',['1792476680198']),
    'new_e':('test',['1792476680319','1792476680578']),
    'new_f':('valid',['1792476680444']),
    'new_g':('test',['1792476680639','1792476680696','1792476680762']),
}


def main():
    p=ROOT/'wheel_slot_local_reviews.json'
    if p.exists(): raise FileExistsError(p)
    review=json.loads((ROOT/'wheel_slot_negative_reviews.json').read_text())
    # Wider supervised targets, applied before training to every positive source.
    for row in review['images']:
        if not row['boxes']: continue
        boxes=np.asarray(row['boxes']); center=(boxes[:,:2]+boxes[:,2:])/2
        half=(boxes[:,2:]-boxes[:,:2])/2*[1.18,1.12]
        row['boxes']=np.c_[center-half,center+half].tolist()
        row['annotation_kind']+=';full_window_margin_18pct_width_12pct_height'
    local=json.loads((ROOT/'results/test_wheel_slots_no_screws/readings.json').read_text())
    old=json.loads((ROOT/'results/test_wheel_slot_model/readings.json').read_text())
    lookup={Path(r['image']).stem:r for r in old['images']}
    assignments={name:(group,split) for group,(split,names) in GROUPS.items() for name in names}
    added=[]; negatives=[]; preview=[]
    for item in local['images']:
        path=Path(item['image'])
        if path.stem not in assignments: continue
        group,split=assignments[path.stem]
        if len(item['slots'])!=5: raise ValueError('Reviewed new layout changed')
        box=item['window_xyxy']; a,b,c,d=math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])
        boxes=[]
        for slot in item['slots']:
            q=np.asarray(slot['quad_in_area']); low=q.min(axis=0); high=q.max(axis=0)
            center=(low+high)/2; half=(high-low)/2*[1.18,1.12]
            boxes.append([*(center-half+[a,b]),*(center+half+[a,b])])
        boxes=np.array(boxes)
        record={'id':'adapt_'+path.stem,'source_image':str(path),'source_sha256':item['source_sha256'],
            'group':'local_adaptation/'+group,'split':split,'count':5,
            'annotation_kind':'assistant_reviewed_complete_wheel_windows_with_margin',
            'crop_xyxy':[a,b,c,d],'boxes':boxes.tolist(),'source':'datas/test_wheel_authorized_adaptation',
            'user_authorized_test_wheel_use':True}
        added.append(record)
        with Image.open(path) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w=rgb.shape[:2]
        bounds=np.array([a,b,c,d]); bounds[:2]=np.minimum(bounds[:2],np.floor(boxes[:,:2].min(axis=0)).astype(int))
        bounds[2:]=np.maximum(bounds[2:],np.ceil(boxes[:,2:].max(axis=0)).astype(int))
        aa,bb,cc,dd=np.clip(bounds,[0,0,0,0],[w,h,w,h]); view=rgb[bb:dd,aa:cc,::-1].copy()
        for i,target in enumerate(boxes-[aa,bb,aa,bb]):
            x1,y1,x2,y2=np.rint(target).astype(int); cv2.rectangle(view,(x1,y1),(x2,y2),(0,255,0),max(1,view.shape[1]//600))
        out=ROOT/'.tmp/slot_adaptation_review'/f"{record['id']}.jpg"; save_image(out,view)
        preview.append((out,f'{group} {split} five full windows'))
        # Mine genuine scene hard negatives only from the newly assigned train groups.
        if split!='train': continue
        for i,slot in enumerate(lookup[path.stem]['slots']):
            proposal=np.array(slot['xyxy_in_area'])+[a,b,a,b]
            size=proposal[2:]-proposal[:2]; proposal+=np.r_[-size*.1,size*.1]
            intersection=np.prod(np.maximum(0,np.minimum(proposal[2:],boxes[:,2:])-np.maximum(proposal[:2],boxes[:,:2])),axis=1)
            if np.any(intersection>0): continue  # Never put a real wheel in a negative crop.
            region=[max(0,math.floor(proposal[0])),max(0,math.floor(proposal[1])),min(w,math.ceil(proposal[2])),min(h,math.ceil(proposal[3]))]
            r={'id':f'adapt_negative_{path.stem}_{i}','source_image':str(path),'source_sha256':item['source_sha256'],
               'group':record['group'],'split':'train','count':0,'negative':True,'crop_xyxy':region,'boxes':[],
               'source':'datas/test_wheel_authorized_adaptation','user_authorized_test_wheel_use':True,
               'annotation_kind':'train_group_only_hard_negative_proposal_no_intersection_with_reviewed_wheels'}
            negatives.append(r)
            x1,y1,x2,y2=region
            out=ROOT/'.tmp/slot_adaptation_negatives'/f"{r['id']}.jpg"; save_image(out,rgb[y1:y2,x1:x2,::-1])
    review['images'].extend(added+negatives)
    review.update(reviewed=False,review_note='User authorized part of test_wheel for training. New A/B/C groups train (7 images), D/F valid (2), E/G test (5); 4 area misses remain unassigned for end-to-end evaluation. All same-meter frames stay together. Wider full-window positive targets; true-scene hard negatives only mined from train groups. This development set is no longer an untouched test set.')
    review['adaptation_groups']={k:{'split':v[0],'images':v[1]} for k,v in GROUPS.items()}
    p.write_text(json.dumps(review,indent=2),encoding='utf-8')
    for start in range(0,len(preview),12): make_overview(preview[start:start+12],ROOT/'.tmp/slot_adaptation_review'/f'overview_{start//12:02}.jpg',columns=3,tile_w=600,tile_h=210)
    preview=[(ROOT/'.tmp/slot_adaptation_negatives'/f"{r['id']}.jpg",'train background, no wheel') for r in negatives]
    make_overview(preview,ROOT/'.tmp/slot_adaptation_negatives/overview.jpg',columns=5,tile_w=200,tile_h=200)
    print('new positives',len(added),'train negatives',len(negatives))


if __name__=='__main__': main()
