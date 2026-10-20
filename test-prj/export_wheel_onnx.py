"""Export pinned wheel models; preserve quantized LeakyReLU via exact 256-value LUTs.

Run in the isolated environment from requirements-wheel-export.txt.
The LUT enumerates every int8 input of each original TFLite unary operator;
it does not use photographs, labels, or learned calibration parameters.
"""
import copy
import hashlib
import json
import os
import subprocess
import sys

import flatbuffers
import numpy as np
import onnx
from onnx import helper, numpy_helper

from prepare_wheel_model import MODELS, ROOT


def leaky_tables(path):
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.lite.python.interpreter import Interpreter
    original = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))
    graph = original.subgraphs[0]
    tables = {}
    for op in graph.operators:
        if original.operatorCodes[op.opcodeIndex].builtinCode != schema.BuiltinOperator.LEAKY_RELU:
            continue
        model = copy.deepcopy(original)
        sub = model.subgraphs[0]
        sub.tensors = [copy.deepcopy(graph.tensors[int(i)]) for i in (op.inputs[0], op.outputs[0])]
        for tensor in sub.tensors:
            tensor.shape = np.array([256], np.int32)
            tensor.shapeSignature = np.array([256], np.int32)
            tensor.buffer = 0
        operator = copy.deepcopy(op)
        operator.inputs, operator.outputs = np.array([0], np.int32), np.array([1], np.int32)
        sub.operators = [operator]
        sub.inputs, sub.outputs = np.array([0], np.int32), np.array([1], np.int32)
        model.subgraphs, model.signatureDefs = [sub], []
        builder = flatbuffers.Builder(1024)
        builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
        interpreter = Interpreter(model_content=bytes(builder.Output()), num_threads=1)
        interpreter.allocate_tensors()
        interpreter.set_tensor(0, np.arange(-128, 128, dtype=np.int8))
        interpreter.invoke()
        tables[graph.tensors[int(op.outputs[0])].name.decode()] = interpreter.get_tensor(1).copy()
    return tables


def preserve_leaky_rounding(model, tables):
    producers = {out: node for node in model.graph.node for out in node.output}
    replacements, removed = {}, set()
    for node in model.graph.node:
        if node.op_type != 'LeakyRelu':
            continue
        dq = producers[node.input[0]]
        consumers = [n for n in model.graph.node if node.output[0] in n.input]
        if len(consumers) != 1:
            raise ValueError('Unexpected LeakyReLU consumers')
        q = consumers[0]
        if dq.op_type != 'DequantizeLinear' or q.op_type != 'QuantizeLinear':
            raise ValueError('Expected quantized LeakyReLU')
        values = tables[q.output[0]]
        name = q.output[0] + '_exact'
        model.graph.initializer.extend([
            numpy_helper.from_array(values, name+'_lut'),
            numpy_helper.from_array(np.array(128, np.int64), name+'_offset')])
        replacements[q.name] = [
            helper.make_node('Cast', [dq.input[0]], [name+'_int'], name=name+'_cast', to=onnx.TensorProto.INT64),
            helper.make_node('Add', [name+'_int', name+'_offset'], [name+'_index'], name=name+'_add'),
            helper.make_node('Gather', [name+'_lut', name+'_index'], list(q.output), name=name+'_lookup', axis=0)]
        removed.add(node.name)
    if len(replacements) != len(tables):
        raise ValueError('Not all TFLite LeakyReLU operators were replaced')
    nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.extend(replacements[node.name])
        elif node.name not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Remove dead dequantizers and their now-unused constants.
    live = {o.name for o in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if live.intersection(node.output):
            kept.append(node)
            live.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [i for i in model.graph.initializer if i.name in live]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return len(replacements)


def main():
    work = ROOT/'.tmp/wheel_export'
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTEROP_THREADS='1',
               TF_NUM_INTRAOP_THREADS='1', TEMP=str(work), TMP=str(work))
    records = []
    for filename, expected in MODELS.items():
        source = ROOT/'models/wheel'/filename
        content = source.read_bytes()
        if hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest() != expected:
            raise ValueError(f'Unexpected source model: {source}')
        target = source.with_suffix('.onnx')
        intermediate = work/target.name
        with (work/(source.stem+'.log')).open('w') as log:
            subprocess.run([sys.executable, '-m', 'tf2onnx.convert', '--opset', '13',
                            '--tflite', str(source), '--output', str(intermediate)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        model = onnx.load(intermediate)
        count = preserve_leaky_rounding(model, leaky_tables(source))
        for tensor in [*model.graph.input, *model.graph.output]:
            tensor.type.tensor_type.shape.dim[0].dim_value = 1
        helper.set_model_props(model, {'source_sha256': hashlib.sha256(content).hexdigest(),
            'input_contract': 'float32 RGB NHWC [1,32,20,3], values 0..255, no /255',
            'quantized_leaky_relu': f'{count} exhaustive int8 lookup tables'})
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, target)
        records.append({'file':target.name, 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'bytes':target.stat().st_size, 'source_sha256':hashlib.sha256(content).hexdigest(),
                        'exact_leaky_relu_tables':count, 'opset':13, 'ir_version':model.ir_version})
        print(f'Exported {target}', flush=True)
    import tensorflow, tf2onnx
    manifest = {'tensorflow':tensorflow.__version__, 'tf2onnx':tf2onnx.__version__,
                'onnx':onnx.__version__, 'models':records,
                'note':'Conversion only; run verify_wheel_onnx.py separately.'}
    (ROOT/'models/wheel/onnx_export.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
