"""Compare ONNX with original LiteRT outputs, using real crops and synthetic probes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from importlib.metadata import version
from PIL import Image

from prepare_wheel_model import ROOT, MODELS
from recognize_wheel import decode_class100, decode_cont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'results/area_wheel_readings/inputs')
    parser.add_argument('--output',type=Path,default=ROOT/'models/wheel/onnx_verification.json')
    parser.add_argument('--readings',type=Path,help='Optional ONNX readings.json for end-to-end comparison')
    parser.add_argument('--baseline',type=Path,default=ROOT/'results/area_wheel_readings/readings.json')
    args = parser.parse_args()
    files = sorted(args.inputs.glob('*.png'))
    if not files:
        parser.error('Real wheel input PNGs required')
    samples = []
    for path in files:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'),np.float32)[None]
        if pixels.shape!=(1,32,20,3):
            raise ValueError(f'Unexpected input: {path}')
        samples.append(('real',path.name,pixels))
    # Every constant level, seeded random RGB, and horizontal/vertical edges.
    for level in range(256):
        samples.append(('synthetic',f'constant_{level}',np.full((1,32,20,3),level,np.float32)))
    rng = np.random.default_rng(20261020)
    for i in range(128):
        samples.append(('synthetic',f'random_{i}',rng.integers(0,256,(1,32,20,3)).astype(np.float32)))
    for axis,size in ((1,32),(2,20)):
        for cut in range(1,size):
            pixels = np.zeros((1,32,20,3),np.float32)
            slices = [slice(None)]*4
            slices[axis] = slice(cut,None)
            pixels[tuple(slices)] = 255
            samples.append(('synthetic',f'edge_{axis}_{cut}',pixels))
    report = {'onnxruntime':ort.__version__,'litert':version('ai-edge-litert'),
              'provider':'CPUExecutionProvider','threads':1,
              'input_contract':'float32 RGB NHWC [1,32,20,3], 0..255, no /255',
              'real_inputs':[{'file':p.name,'sha256':sha(p)} for p in files],
              'synthetic_seed':20261020,'models':{}}
    passed = True
    for kind,filename in zip(('class100','cont'),MODELS):
        source = ROOT/'models/wheel'/filename
        target = source.with_suffix('.onnx')
        graph = onnx.load(target)
        onnx.checker.check_model(graph,full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
        native = Interpreter(model_path=str(source),num_threads=1)
        native.allocate_tensors()
        inp,out = native.get_input_details()[0]['index'],native.get_output_details()[0]['index']
        decode = decode_class100 if kind=='class100' else decode_cont
        results = {group:{'count':0,'identical_outputs':0,'same_top_class':0,'same_digit':0,'max_abs':0.,'differences':[]}
                   for group in ('real','synthetic')}
        for group,name,pixels in samples:
            native.set_tensor(inp,pixels)
            native.invoke()
            a = native.get_tensor(out)[0]
            b = session.run(None,{session.get_inputs()[0].name:pixels})[0][0]
            exact = bool(np.array_equal(a,b))
            same_digit = decode(a)['digit']==decode(b)['digit']
            error = float(np.max(np.abs(a-b)))
            row = results[group]
            row['count'] += 1
            row['identical_outputs'] += int(exact)
            row['same_top_class'] += int(np.argmax(a)==np.argmax(b))
            row['same_digit'] += int(same_digit)
            row['max_abs'] = max(row['max_abs'],error)
            if not exact:
                row['differences'].append({'input':name,'max_abs':error,'tflite_digit':decode(a)['digit'],
                                           'onnx_digit':decode(b)['digit']})
        passed &= all(r['identical_outputs']==r['count'] for r in results.values())
        report['models'][kind] = {'onnx_file':target.name,'onnx_sha256':sha(target),
             'tflite_sha256':sha(source),'bytes':target.stat().st_size,'opset':graph.opset_import[0].version,
             'ir_version':graph.ir_version,'operators':sorted({n.op_type for n in graph.graph.node}),
             'inputs':[(i.name,i.shape,i.type) for i in session.get_inputs()],
             'outputs':[(o.name,o.shape,o.type) for o in session.get_outputs()],**results}
        print(kind,json.dumps(results),flush=True)
    report['all_real_outputs_exact'] = all(m['real']['identical_outputs']==len(files) for m in report['models'].values())
    report['all_verified_outputs_exact'] = passed
    if args.readings:
        baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
        current = json.loads(args.readings.read_text(encoding='utf-8'))
        before = {r['source_sha256']:r for r in baseline['images']}
        after = {r['source_sha256']:r for r in current['images']}
        if len(before)!=len(baseline['images']) or len(after)!=len(current['images']):
            raise ValueError('Duplicate source hashes')
        differences, slots = [],0
        if before.keys()!=after.keys():
            differences.append('source image sets differ')
        for key in before.keys() & after.keys():
            a,b = before[key],after[key]
            fields = ('status','window_xyxy','layout','reasons','class100_raw_digits',
                      'cont_raw_digits','class100_reading_candidate','cont_reading_candidate','models_agree')
            if any(a.get(f)!=b.get(f) for f in fields) or len(a['slots'])!=len(b['slots']):
                differences.append(a['image'])
                continue
            for sa,sb in zip(a['slots'],b['slots']):
                slots += 1
                equal = sa['quad_in_area']==sb['quad_in_area'] and sha(Path(sa['model_input']))==sha(Path(sb['model_input']))
                for kind in ('class100','cont'):
                    va = {k:v for k,v in sa[kind].items() if k!='inference_ms'}
                    vb = {k:v for k,v in sb[kind].items() if k!='inference_ms'}
                    equal &= va==vb
                if not equal:
                    differences.append(f"{a['image']} slot {sa['index']}")
        report['end_to_end'] = {'baseline_sha256':sha(args.baseline),'readings_sha256':sha(args.readings),
                                'images':len(after),'compared_slots':slots,'differences':differences}
        passed &= not differences
        print('end_to_end',json.dumps(report['end_to_end']),flush=True)
    report['passed'] = bool(passed)
    report['note'] = 'Finite CPU comparison; not a proof for every input or another provider/device. Synthetic probes are not accuracy evaluation.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not passed:
        raise SystemExit('ONNX comparison failed; inspect verification report')


if __name__=='__main__':
    main()
