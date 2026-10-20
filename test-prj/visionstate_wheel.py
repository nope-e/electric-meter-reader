"""Pinned VisionState ONNX reader; official preprocessing and joint decoding.

Uses Apache-2.0 source from oleost/VisionState; attribution in vendor/visionstate.
"""
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from main import ROOT
from vendor.visionstate.common import cell_input, decode_shifted

MODEL_SHA256 = '37912f885b061a7b12298872fbdfa1b73ca0664363b293a55617b1d3680a98ac'


def decode_logits(logits):
    logits = np.asarray(logits, np.float64)
    if logits.ndim!=2 or logits.shape[1]!=100 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected nonempty wheels x 100 finite logits')
    exponential = np.exp(logits-logits.max(axis=1,keepdims=True))
    probabilities = exponential/exponential.sum(axis=1,keepdims=True)
    # Author's release defaults from settings.py at the pinned code revision.
    near = sum(np.roll(probabilities,s,axis=1) for s in (-1,0,1))
    logp = np.log(np.maximum(near,1e-12))
    value,score,shift = decode_shifted(logp,max_shift=2,penalty=.3)
    digits = len(logits)
    raw_bins = np.argmax(logits,axis=1)
    return {'independent_raw_digits':''.join(str(int(b)//10) for b in raw_bins),
            'independent_states':(raw_bins/10).tolist(),
            'joint_value_in_last_wheel_units':value,'joint_log_score':score,
            'common_shift_bins':shift,
            'raw_digits':f'{math.floor(value)%10**digits:0{digits}d}',
            'author_rounded_raw_digits':f'{math.floor(value+.5)%10**digits:0{digits}d}',
            'logits':logits.tolist(),'probabilities':probabilities.tolist()}


class VisionStateWheel:
    def __init__(self, model=ROOT/'models/visionstate/wheels-v1.onnx'):
        self.path = Path(model)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256!=MODEL_SHA256:
            raise ValueError('Unexpected VisionState model hash')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])
        inp,out = self.session.get_inputs()[0],self.session.get_outputs()[0]
        if inp.shape[1:]!=[1,48,32] or inp.type!='tensor(float)' or out.shape[1:]!=[100]:
            raise ValueError('Unexpected VisionState input/output contract')
        self.input_name = inp.name
        self.session.run(None,{inp.name:np.zeros((1,1,48,32),np.float32)})

    def predict(self,crops):
        if not crops:
            raise ValueError('At least one correctly ordered wheel crop required')
        cells = np.stack([cell_input(c.convert('RGB') if isinstance(c,Image.Image) else Image.fromarray(c)) for c in crops])
        start = time.perf_counter()
        logits = self.session.run(None,{self.input_name:cells[:,None]})[0]
        ms = (time.perf_counter()-start)*1000
        result = decode_logits(logits)
        result['inference_ms'] = ms
        return result,cells
