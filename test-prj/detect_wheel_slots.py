"""ONNX wheel-slot detector. Detects a variable number of complete wheel windows."""
import ast
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from main import ROOT,preprocess,decode


def ordered_slot_crops(rgb,detections,margin_x=0.,margin_y=0.):
    """Use every nonempty post-NMS box, left to right; never extrapolate a wheel."""
    h,w=rgb.shape[:2]
    if margin_x<0 or margin_y<0: raise ValueError('Negative slot margin')
    slots=[]
    for detection in sorted(detections,key=lambda d:((d['xyxy'][0]+d['xyxy'][2])/2,(d['xyxy'][1]+d['xyxy'][3])/2)):
        box=np.asarray(detection['xyxy'],float)
        if box.shape!=(4,) or not np.isfinite(box).all(): raise ValueError('Invalid slot box')
        padding=(box[2:]-box[:2])*[margin_x,margin_y]
        a,b=np.maximum([0,0],np.floor(box[:2]-padding).astype(int))
        c,d=np.minimum([w,h],np.ceil(box[2:]+padding).astype(int))
        if c<=a or d<=b: continue
        slots.append({'xyxy':box.tolist(),'crop_xyxy':[int(a),int(b),int(c),int(d)],
            'score':float(detection['score']),'pixels':rgb[b:d,a:c].copy()})
    return slots


class WheelSlotDetector:
    def __init__(self,model=ROOT/'models/wheel_slots/wheel_slots.onnx',threads=4,confidence=.25,iou=.45):
        if threads<1 or not 0<confidence<1 or not 0<iou<1: raise ValueError('Invalid detector settings')
        self.path=Path(model); self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0]
        if self.input.shape!=[1,3,640,640] or self.input.type!='tensor(float)': raise ValueError('Unexpected slot input')
        names=ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        self.names=dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
        if self.names!={0:'wheel_slot'}: raise ValueError('Unexpected slot classes')
        self.confidence,self.iou=confidence,iou
        self.session.run(None,{self.input.name:np.zeros(self.input.shape,np.float32)})

    def predict(self,rgb):
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        tensor,scale,left,top=preprocess(bgr,640,640)
        started=time.perf_counter(); output=self.session.run(None,{self.input.name:tensor})[0]
        ms=(time.perf_counter()-started)*1000
        detections,best=decode(output,scale,left,top,rgb.shape,self.names,self.confidence,self.iou)
        slots=ordered_slot_crops(rgb,detections)
        return slots,{'inference_ms':ms,'best_raw_score':best,'count':len(slots)}
