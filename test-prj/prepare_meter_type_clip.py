"""Prepare a zero-shot MobileCLIP LCD/wheel classifier for ONNX Runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOSITORY = "Xenova/mobileclip_s0"
PROMPTS = {
    "lcd": [
        "a photo of an electricity meter with a digital LCD display.",
        "a close-up of an electric meter with seven-segment digits on a liquid crystal screen.",
        "a digital electricity meter with black segmented numbers on a gray LCD screen.",
        "a digital kilowatt-hour meter with an electronic numeric display.",
    ],
    "wheel": [
        "a photo of an electricity meter with a mechanical rolling digit counter.",
        "a close-up of an electric meter with rotating numbered wheels.",
        "an electricity meter with white printed numbers on black mechanical drums.",
        "a kilowatt-hour meter with a mechanical odometer counter.",
    ],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def download_files(directory):
    manifest_path = directory / "source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        info = fetch_json(f"https://huggingface.co/api/models/{REPOSITORY}?blobs=true")
        siblings = {item["rfilename"]: item for item in info["siblings"]}
        names = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "onnx/text_model_quantized.onnx", "onnx/vision_model.onnx", "onnx/vision_model_quantized.onnx"]
        manifest = {"repository": REPOSITORY, "revision": info["sha"], "files": [
            {"name": name, "expected_sha256": siblings[name].get("lfs", {}).get("sha256"),
             "git_blob_id": siblings[name].get("blobId")} for name in names]}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for item in manifest["files"]:
        name = item["name"]
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("expected_sha256")
        if target.exists() and (not expected or sha256(target) == expected):
            item["sha256"] = sha256(target)
            item["bytes"] = target.stat().st_size
            continue
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{manifest['revision']}/{name}"
        print(f"Downloading {name}", flush=True)
        partial = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        digest = sha256(partial)
        if expected and digest != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
        if not expected and item.get("git_blob_id"):
            data = partial.read_bytes()
            blob_id = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob_id != item["git_blob_id"]:
                raise ValueError(f"Git blob mismatch: {name}")
        partial.replace(target)
        item.update(sha256=digest, bytes=target.stat().st_size)
        print(f"Verified {name}: {item['bytes']} bytes", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_classifier(directory, variant, class_vectors):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    source = "vision_model.onnx" if variant == "fp32" else "vision_model_quantized.onnx"
    model = onnx.load(directory / "onnx" / source)
    embedding = model.graph.output[0].name
    model.graph.initializer.extend([
        numpy_helper.from_array(class_vectors.T.astype(np.float32), "meter_class_vectors"),
        numpy_helper.from_array(np.array(100, dtype=np.float32), "meter_logit_scale"),
    ])
    model.graph.node.extend([
        helper.make_node("LpNormalization", [embedding], ["meter_normalized"], axis=-1, p=2),
        helper.make_node("MatMul", ["meter_normalized", "meter_class_vectors"], ["cosine_scores"]),
        helper.make_node("Mul", ["cosine_scores", "meter_logit_scale"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["class_scores"], axis=-1),
    ])
    del model.graph.output[:]
    model.graph.output.extend([
        helper.make_tensor_value_info("cosine_scores", TensorProto.FLOAT, ["batch_size", 2]),
        helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, ["batch_size", 2]),
    ])
    for key, value in {"class_names": ["lcd", "wheel"], "prompts": PROMPTS,
                       "score_note": "Closed-set zero-shot similarity, not calibrated correctness probability.",
                       "preprocessing": "RGB, PIL bilinear shortest-edge resize 256, center crop 256, NCHW float32 /255; no mean/std normalization."}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    onnx.checker.check_model(model)
    output = directory / f"meter_type_{variant}.onnx"
    onnx.save(model, output)
    return {"file": str(output), "sha256": sha256(output), "bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "models/mobileclip_s0"
    directory.mkdir(parents=True, exist_ok=True)
    for key, relative in {"HF_HOME": ".cache/huggingface", "TEMP": ".tmp", "TMP": ".tmp"}.items():
        os.environ[key] = str(ROOT / relative)
    manifest = download_files(directory)
    if args.download_only:
        return
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=0, pad_token="[PAD]")
    prompts = [prompt for group in PROMPTS.values() for prompt in group]
    encoded = tokenizer.encode_batch(prompts)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(directory / "onnx/text_model_quantized.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    available = {"input_ids": np.array([item.ids for item in encoded], np.int64),
                 "attention_mask": np.array([item.attention_mask for item in encoded], np.int64)}
    text = session.run(None, {i.name: available[i.name] for i in session.get_inputs()})[0]
    text = text / np.linalg.norm(text, axis=-1, keepdims=True)
    vectors = np.stack([group.mean(axis=0) for group in np.split(text, 2)])
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    np.save(directory / "class_vectors.npy", vectors)
    manifest.update(prompts=PROMPTS, text_encoder="text_model_quantized.onnx", class_names=list(PROMPTS),
                    classifiers=[build_classifier(directory, variant, vectors) for variant in ["fp32", "int8"]])
    (directory / "classifier.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["classifiers"], indent=2), flush=True)


if __name__ == "__main__":
    main()
