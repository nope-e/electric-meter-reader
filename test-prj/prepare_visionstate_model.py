"""Fetch the pinned ONNX wheel reader; verify SHA256 before use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from main import ROOT
from visionstate_wheel import MODEL_SHA256

REVISION = '42f32402973b3bd3de0258766404232b11e8216f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',choices=['huggingface.co','hf-mirror.com'],default='huggingface.co')
    args = parser.parse_args()
    path = ROOT/'models/visionstate/wheels-v1.onnx'
    url = f'https://{args.host}/oleost/visionstate-wheel-reader/resolve/{REVISION}/wheels-v1.onnx'
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(url,timeout=120) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest()!=MODEL_SHA256:
        raise ValueError('Unexpected VisionState model hash')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    manifest_path = path.with_name('source.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(repository='https://huggingface.co/oleost/visionstate-wheel-reader',revision=REVISION,
        file=path.name,sha256=MODEL_SHA256,bytes=len(content),license='Apache-2.0',
        verified_source_url=url)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(path,MODEL_SHA256)


if __name__=='__main__':
    main()
