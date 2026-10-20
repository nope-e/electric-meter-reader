"""Fetch pinned AI-on-the-edge single-wheel models and verify their Git blobs."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = 'cdc8450924e9338ab564c3fd76ad998e7b7cbb84'
MODELS = {'dig-class100-0182-s2_q.tflite':'3073ecb0be4c3b89b7a601a9a82370183b895211',
          'dig-cont_0900_s3_q.tflite':'f0d712738e06ac06919804afe776bf1e95554d20'}


def main():
    destination = ROOT/'models/wheel'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected_blob in MODELS.items():
        path = destination/name
        url = f'https://raw.githubusercontent.com/jomjol/AI-on-the-edge-device/{COMMIT}/sd-card/config/{name}'
        cached = ROOT/'.tmp'/name
        content = path.read_bytes() if path.exists() else (
            cached.read_bytes() if cached.exists() else urllib.request.urlopen(url, timeout=120).read())
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        if blob != expected_blob:
            raise ValueError(f'Unexpected model content: {name} {blob}')
        if not path.exists():
            path.write_bytes(content)
        records.append({'file':name,'source_url':url,'git_blob_sha1':blob,
                        'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    manifest = {'repository':'https://github.com/jomjol/AI-on-the-edge-device',
                'commit':COMMIT,'models':records,
                'license_url':f'https://github.com/jomjol/AI-on-the-edge-device/blob/{COMMIT}/Licence.md',
                'input':'float32 RGB NHWC [1,32,20,3], pixel values 0..255 (no /255)',
                'source_format':'TFLite; exported ONNX provenance and checks are stored separately in onnx_export.json and onnx_verification.json'}
    (destination/'source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
