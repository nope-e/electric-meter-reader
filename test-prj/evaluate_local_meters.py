"""Run and score all user photos with the current ONNX pipeline and fixed references."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from main import ROOT,make_overview


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reading(raw,places):
    if raw is None:return None
    return raw[:-places]+'.'+raw[-places:] if places else raw


def summarize(rows):
    scored=[r for r in rows if r['exact_match'] is not None]
    correct=sum(r['exact_match'] for r in scored)
    return {'images':len(rows),'type_correct':sum(r['type_correct'] for r in rows),
            'with_area':sum(r['has_area'] for r in rows),
            'with_reading':sum(r['candidate'] is not None for r in rows),
            'scorable':len(scored),'correct':correct,'incorrect':len(scored)-correct,
            'unscored':len(rows)-len(scored),
            'end_to_end_exact_match_rate':correct/len(scored) if scored else None,
            'verified_correct_fraction_of_all_images':correct/len(rows) if rows else None,
            'failure_causes':dict(Counter(r['failure_cause'] for r in scored if not r['exact_match']))}


def collect_inventory():
    rows=[]
    for folder in ['lcd','wheel','test_wheel']:
        for path in sorted((ROOT.parent/'datas'/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
                rows.append({'image':str(path.resolve()),'source_folder':folder,
                             'reference_type':'digital' if folder=='lcd' else 'wheel',
                             'source_sha256':sha(path)})
    if not rows:raise ValueError('No user photos')
    if len({r['source_sha256'] for r in rows})!=len(rows):
        raise ValueError('Duplicate source files: review inventory before scoring')
    return rows


def run_pipeline(output):
    if output.exists():raise FileExistsError('Use a new output or --score-existing')
    output.mkdir(parents=True)
    inventory=collect_inventory()
    (output/'inventory.json').write_text(json.dumps({'images':inventory},indent=2),encoding='utf-8')
    commands=[
        ['detect_meter_area.py','--input',str(ROOT.parent/'datas/lcd'),str(ROOT.parent/'datas/wheel'),str(ROOT.parent/'datas/test_wheel'),'--output',str(output/'area')],
        ['recognize.py','--windows',str(output/'area/detections.json'),'--meter-type','digital','--output',str(output/'digital')],
        ['recognize_wheel_slots.py','--windows',str(output/'area/detections.json'),'--output',str(output/'wheel')],
    ]
    for index,command in enumerate(commands):
        with (output/f'pipeline_{index}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def evaluate(output,reference_path):
    inventory=read_json(output/'inventory.json')['images']
    reference=read_json(reference_path)
    area_document=read_json(output/'area/detections.json')
    digital_document=read_json(output/'digital/readings.json')
    wheel_document=read_json(output/'wheel/readings.json')
    area={str(Path(r['image']).resolve()):r for r in area_document['images']}
    digital={str(Path(r['image']).resolve()):r for r in digital_document['images']}
    wheel={str(Path(r['image']).resolve()):r for r in wheel_document['images']}
    if set(area)!={str(Path(r['image']).resolve()) for r in inventory}:raise ValueError('Pipeline did not cover inventory')
    classifier_train={r['sha256'] for r in read_json(ROOT/'models/meter_type_cnn/training.json')['training_images']}
    slot_manifest=read_json(ROOT/'datasets/wheel_slots_local/manifest.json')
    slot_splits={}
    for record in slot_manifest:
        digest=record['source_sha256']
        if digest in slot_splits and slot_splits[digest]!=record['split']:raise ValueError('Slot source split conflict')
        slot_splits[digest]=record['split']
    area_train={r['image_sha256'] for r in read_json(ROOT/'datasets/meter_area/manifest.json') if r['split']=='train'}
    rows=[]
    for item in inventory:
        path=Path(item['image']).resolve(); key=path.relative_to(ROOT.parent/'datas').as_posix()
        digest=sha(path)
        if digest!=item['source_sha256']:raise ValueError('Source changed: '+key)
        label=reference['labels'].get(key)
        if label is None:raise ValueError('Reference must explicitly include or exclude: '+key)
        if label['source_sha256']!=digest:raise ValueError('Reference hash mismatch: '+key)
        window=area[str(path)]
        if window['source_sha256']!=digest:raise ValueError('Area source hash mismatch')
        predicted_type=window['meter_type']; expected_type=item['reference_type']
        if label['meter_type']!=expected_type:raise ValueError('Reference type conflict')
        prediction=(digital if predicted_type=='digital' else wheel)[str(path)]
        raw=prediction.get('raw_digits') or None
        candidate=prediction.get('reading') if predicted_type=='digital' else prediction.get('visionstate_reading_candidate')
        expected=label['raw_digits']; has_area=bool(window['detections'])
        exact=(predicted_type==expected_type and has_area and candidate is not None and raw==expected) if expected is not None else None
        cause=None
        if exact is False:
            if predicted_type!=expected_type:cause='type_misclassification'
            elif not has_area:cause='no_area'
            elif candidate is None:cause='no_reading'
            elif predicted_type=='wheel' and len(prediction['slots'])!=len(expected):cause='wheel_slot_count_mismatch'
            elif predicted_type=='digital' and len(raw)!=len(expected):cause='lcd_digit_count_mismatch'
            else:cause='digit_value_mismatch'
        row={'image':str(path),'relative_image':key,'source_sha256':digest,'source_folder':item['source_folder'],
             'reference_type':expected_type,'predicted_type':predicted_type,'type_correct':predicted_type==expected_type,
             'has_area':has_area,'raw_digits':raw,'candidate':candidate,'reference_raw':expected,
             'reference_reading':reading(expected,label['decimal_places']),'exact_match':exact,
             'failure_cause':cause,'unscored_reason':label.get('reason') if expected is None else None,
             'classifier_training_image':digest in classifier_train,'area_training_image':digest in area_train,
             'wheel_slot_dataset_split':slot_splits.get(digest) if expected_type=='wheel' else None,
             'wheel_slot_count':len(prediction['slots']) if predicted_type=='wheel' else None,
             'preview':prediction.get('annotated') or window['annotated']}
        rows.append(row)
    summary={'overall':summarize(rows),
             'by_type':{t:summarize([r for r in rows if r['reference_type']==t]) for t in ['digital','wheel']},
             'by_folder':{f:summarize([r for r in rows if r['source_folder']==f]) for f in ['lcd','wheel','test_wheel']},
             'training_exposure':{'classifier_training_images':sum(r['classifier_training_image'] for r in rows),
                 'area_training_images':sum(r['area_training_image'] for r in rows),
                 'wheel_slot_dataset_splits':dict(Counter(r['wheel_slot_dataset_split'] or 'unused' for r in rows if r['reference_type']=='wheel'))},
             'metrics_note':'Exact full digit string with leading zeros retained and correct type routing. No-area and no-reading failures stay in the scorable denominator. Unknown references excluded only from exact-match rate; all photos remain in inventory, previews and verified-correct/all lower bound. Failure cause is the first observable mismatch, not a proven root cause.',
             'reference_note':reference['source'],
             'evaluation_kind':'Local-photo replay including training photos and repeated views; not an independent generalization benchmark.',
             'reference_sha256':sha(reference_path),
             'model_sha256':{'classifier':area_document['summary']['classifier_model_sha256'],
                 'area':area_document['summary']['area_model_sha256'],
                 'lcd_reader':digital_document['summary']['model_sha256'],
                 'wheel_slots':wheel_document['summary']['slot_model_sha256'],
                 'wheel_reader':wheel_document['summary']['recognizer_sha256']}}
    models={'classifier':'models/meter_type_cnn/meter_type.onnx','area':'models/meter_area/area.onnx',
            'lcd_reader':'models/digits.onnx','wheel_slots':'models/wheel_slots/wheel_slots.onnx',
            'wheel_reader':'models/visionstate/wheels-v1.onnx'}
    for name,path in models.items():
        if sha(ROOT/path)!=summary['model_sha256'][name]:raise ValueError('Model changed since inference: '+name)
    (output/'reference_snapshot.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'evaluation.json').write_text(json.dumps({'summary':summary,'images':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    fields=[k for k in rows[0] if k!='preview']
    with (output/'evaluation.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    for name,selected in [('all_images',rows),('failures',[r for r in rows if r['exact_match'] is False]),('unscored',[r for r in rows if r['exact_match'] is None])]:
        for start in range(0,len(selected),12):
            batch=selected[start:start+12]
            make_overview([(Path(r['preview']),f"{Path(r['image']).stem} pred={r['raw_digits'] or 'none'} ref={r['reference_raw'] or 'unknown'}") for r in batch],
                          output/f'{name}_{start//12:02}.jpg',columns=3,tile_w=640,tile_h=220)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for row in rows:
        if row['exact_match'] is False:print('FAIL',row['relative_image'],row['candidate'],'expected',row['reference_reading'],row['failure_cause'])
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/all_local_meters')
    parser.add_argument('--reference',type=Path,default=ROOT/'local_meter_reference.json')
    parser.add_argument('--score-existing',action='store_true',help='Score a completed run without changing models or running inference again')
    args=parser.parse_args()
    if not args.score_existing:run_pipeline(args.output.resolve())
    evaluate(args.output.resolve(),args.reference)


if __name__=='__main__':main()
