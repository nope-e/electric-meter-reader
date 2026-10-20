"""Whole-image digital/wheel classification plus independent area-only ONNX detection."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from classify_meter import MeterTypeClassifier
from main import ROOT, decode, make_overview, preprocess, save_image


SELECTION_RULES = {
    'confidence': 'highest_confidence_area_tiebreak',
    'largest_area': 'largest_area_in_original_image_score_tiebreak',
}


def select_area(candidates, selection='confidence'):
    """Rank post-NMS candidates; return the original candidate without modifying it."""
    if selection not in SELECTION_RULES:
        raise ValueError(f'Unknown area selection rule: {selection}')

    def rank(candidate):
        x1, y1, x2, y2 = candidate['xyxy']
        area = (x2-x1)*(y2-y1)
        score = candidate['score']
        return (score, area) if selection == 'confidence' else (area, score)

    return max(candidates, key=rank, default=None)


class MeterAreaPipeline:
    def __init__(self, area_model=ROOT/'models/meter_area/area.onnx',
                 classifier_model=ROOT/'models/meter_type_cnn/meter_type.onnx',
                 threads=4, confidence=.25, iou=.45, collapse_types=False,
                 selection='confidence'):
        if selection not in SELECTION_RULES:
            raise ValueError(f'Unknown area selection rule: {selection}')
        self.selection = selection
        self.classifier = MeterTypeClassifier(classifier_model, threads)
        self.confidence, self.iou = confidence, iou
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(area_model), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        if self.input.shape != [1,3,640,640]:
            raise ValueError(f'Expected fixed [1,3,640,640], got {self.input.shape}')
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        self.names = dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
        if self.names != {0:'area'} and not (collapse_types and self.names == {0:'lcd',1:'wheel'}):
            raise ValueError(f'Expected area-only model, got {self.names}; use --collapse-types only for old-model comparison')
        self.session.run(None, {self.input.name:np.zeros(self.input.shape, dtype=np.float32)})

    def predict(self, image):
        """Input is orientation-corrected PIL RGB pixels; paths/labels never enter prediction."""
        image = image.convert('RGB')
        classified, _ = self.classifier.classify(image)
        meter_type = {'lcd':'digital','digital':'digital','wheel':'wheel'}[classified['meter_type']]
        bgr = np.asarray(image)[:, :, ::-1].copy()
        tensor, scale, left, top = preprocess(bgr, 640, 640)
        started = time.perf_counter()
        output = self.session.run(None, {self.input.name:tensor})[0]
        detection_ms = (time.perf_counter()-started)*1000
        candidates, raw_score = decode(output, scale, left, top, bgr.shape, self.names,
                                       self.confidence, self.iou, agnostic=True)
        # The comparison path ignores old window types; geometry and scores remain intact.
        for candidate in candidates:
            candidate['source_detector_class'] = candidate['class_name']
            candidate.update(class_name='area', class_id=0)
        selected = select_area(candidates, self.selection)
        return {'meter_type':meter_type,
                'type_scores':{('digital' if k == 'lcd' else k):v
                               for k,v in classified['relative_class_scores'].items()},
                'type_inference_ms':classified['inference_ms'], 'area_inference_ms':detection_ms,
                'candidate_count':len(candidates), 'best_raw_score':raw_score,
                'selection':SELECTION_RULES[self.selection],
                'detections':[selected.copy()] if selected else [], 'candidates':candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area-model', type=Path, default=ROOT/'models/meter_area/area.onnx')
    parser.add_argument('--classifier-model', type=Path, default=ROOT/'models/meter_type_cnn/meter_type.onnx')
    parser.add_argument('--input', type=Path, nargs='+', default=[ROOT.parent/'datas/lcd',ROOT.parent/'datas/wheel'])
    parser.add_argument('--output', type=Path, default=ROOT/'results/meter_area_confidence')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--collapse-types', action='store_true')
    parser.add_argument('--selection', choices=SELECTION_RULES, default='confidence',
                        help='Post-NMS ranking: confidence first (default), or legacy largest area')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Choose a new --output; destination exists: {args.output}')
    if args.threads < 1 or not 0 < args.conf < 1 or not 0 < args.iou < 1:
        parser.error('Invalid threads/conf/iou')
    extensions = {'.jpg','.jpeg','.png','.webp','.bmp'}
    paths = sorted({p.resolve() for item in args.input
                    for p in (item.rglob('*') if item.is_dir() else [item])
                    if p.is_file() and p.suffix.lower() in extensions})
    if not paths:
        parser.error('No images found')
    pipeline = MeterAreaPipeline(args.area_model, args.classifier_model, args.threads,
                                 args.conf, args.iou, args.collapse_types, args.selection)
    rows = []
    for path in paths:
        started = time.perf_counter()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        row = pipeline.predict(image)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = f'{path.stem}_{digest[:8]}'
        reference_type = {'lcd':'digital','wheel':'wheel'}.get(path.parent.name)
        row.update(image=str(path), source_sha256=digest, reference_type=reference_type,
                   width=image.width, height=image.height,
                   decode_and_inference_ms=(time.perf_counter()-started)*1000)
        preview = image.copy()
        preview.thumbnail((960,1080), Image.Resampling.BILINEAR)
        annotated = np.asarray(preview)[:, :, ::-1].copy()
        sx, sy = preview.width/image.width, preview.height/image.height
        for candidate in row['candidates']:
            a,b,c,d = candidate['xyxy']
            cv2.rectangle(annotated, (round(a*sx),round(b*sy)), (round(c*sx),round(d*sy)), (150,150,150), 1)
        for detection in row['detections']:
            a,b,c,d = detection['xyxy']
            cv2.rectangle(annotated, (round(a*sx),round(b*sy)), (round(c*sx),round(d*sy)), (30,240,100), 2)
            crop = image.crop((int(np.floor(a)),int(np.floor(b)),int(np.ceil(c)),int(np.ceil(d))))
            crop_path = args.output/'crops'/f'{artifact}.jpg'
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path, quality=95)
            detection['crop'] = str(crop_path)
        text = f"{row['meter_type']}  area={row['detections'][0]['score']:.3f}" if row['detections'] else f"{row['meter_type']}  no area"
        cv2.rectangle(annotated, (0,0), (annotated.shape[1],36), (20,20,20), -1)
        cv2.putText(annotated, text, (8,25), cv2.FONT_HERSHEY_SIMPLEX, .65, (100,240,100), 2)
        annotated_path = args.output/'annotated'/f'{artifact}.jpg'
        save_image(annotated_path, annotated)
        row['annotated'] = str(annotated_path)
        rows.append(row)
        print(f'{path.name}: {text} candidates={row["candidate_count"]}', flush=True)
    labeled = [r for r in rows if r['reference_type']]
    summary = {'image_count':len(rows), 'images_with_area':sum(bool(r['detections']) for r in rows),
               'type_matches':sum(r['meter_type']==r['reference_type'] for r in labeled),
               'type_reference_count':len(labeled),
               'by_reference_type':{name:{'images':sum(r['reference_type']==name for r in rows),
                                          'with_area':sum(r['reference_type']==name and bool(r['detections']) for r in rows),
                                          'type_matches':sum(r['reference_type']==name and r['meter_type']==name for r in rows)}
                                    for name in ['digital','wheel']},
               'predicted_types':dict(Counter(r['meter_type'] for r in rows)),
               'area_model':str(args.area_model.resolve()), 'classifier_model':str(args.classifier_model.resolve()),
               'area_model_sha256':hashlib.sha256(args.area_model.read_bytes()).hexdigest(),
               'classifier_model_sha256':hashlib.sha256(args.classifier_model.read_bytes()).hexdigest(),
               'detector_names':pipeline.names, 'collapse_types':args.collapse_types,
               'confidence':args.conf, 'nms_iou':args.iou,
               'selection':SELECTION_RULES[args.selection],
               'median_area_inference_ms':float(np.median([r['area_inference_ms'] for r in rows])),
               'median_type_inference_ms':float(np.median([r['type_inference_ms'] for r in rows])),
               'limitations':'Classifier trained on these local photos; type matches are a fit check. Area counts are not spatial accuracy; no local ground-truth boxes. Desktop CPU, no phone validation.'}
    (args.output/'detections.json').write_text(json.dumps({'summary':summary,'images':rows}, indent=2), encoding='utf-8')
    for name in ['digital','wheel',None]:
        group = [r for r in rows if r['reference_type']==name]
        for start in range(0,len(group),12):
            selected = group[start:start+12]
            make_overview([(Path(r['annotated']),f"type={r['meter_type']} area={bool(r['detections'])}") for r in selected],
                          args.output/f'overview_{name or "unlabeled"}_{start//12:02}.jpg', columns=3, tile_w=480, tile_h=400)
        crops = [(Path(r['detections'][0]['crop']),r['meter_type']) for r in group if r['detections']]
        for start in range(0,len(crops),12):
            make_overview(crops[start:start+12], args.output/f'areas_{name or "unlabeled"}_{start//12:02}.jpg',
                          columns=3, tile_w=640, tile_h=220)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
