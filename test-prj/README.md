# 电表类型分类、LCD 窗口检测与数字识别实验

LCD 窗口定位和数字识别使用现成模型；新增的 LCD / 滚轮表分类使用本地照片微调 MobileNetV3-small。LCD 原图输入默认为 `../datas/lcd`，类型分类默认读取 `../datas`。

实测结论、误检案例及运行边界见 [TEST_RESULTS.md](TEST_RESULTS.md)。
数字识别结果及人工对照说明见 [RECOGNITION_RESULTS.md](RECOGNITION_RESULTS.md)。
LCD 画质/可读性探索见 [QUALITY_RESULTS.md](QUALITY_RESULTS.md)。
类型分类的全量训练结果与历史分组验证见 [METER_TYPE_RESULTS.md](METER_TYPE_RESULTS.md)。

## LCD / 滚轮表分类

已生成 `models/meter_type_cnn/meter_type.onnx`，运行阶段仅用 ONNX Runtime，输入整张照片，不依赖文件名或目录标签：

```powershell
uv sync --cache-dir .cache\uv
.\.venv\Scripts\python.exe classify_meter.py
.\.venv\Scripts\python.exe classify_meter.py --input ..\datas\wheel\1792416424922.jpg --output results\meter_type_single
```

输出 `results/meter_type/overview.jpg`、`classifications.json`、`classifications.csv`、带分类标签的原图预览与模型实际输入 PNG。类别为 `lcd` / `wheel`，相对分数未经校准；当前没有其他物体拒识。

Python 中可复用一个会话：

```python
from PIL import Image, ImageOps
from classify_meter import MeterTypeClassifier

classifier = MeterTypeClassifier()
with Image.open("photo.jpg") as source:
    result, model_input = classifier.classify(ImageOps.exif_transpose(source).convert("RGB"))
print(result["meter_type"])
```

按用户确认的固定表型范围，默认使用**全部 86 张图训练**的最终模型：33 张 LCD、53 张滚轮图，两种滚轮表型的照片都包含在训练中。该 ONNX 原图回测 **86/86**（LCD 33/33、滚轮 53/53），结果总览在 `results/meter_type/overview.jpg`，属于训练集回测。

此前按表型整组留出的 75/86 是额外的跨表型探索，所用 `fold0.onnx` / `fold1.onnx` 不用于默认分类；默认模型为 `meter_type.onnx`，没有留出任何照片。尚未接入自动分类后读数流程。

复现训练、导出与原图一致性检查：

```powershell
uv sync --group export --cache-dir .cache\uv
.\.venv\Scripts\python.exe train_meter_type.py
.\.venv\Scripts\python.exe verify_meter_type.py
```

首次训练会把 torchvision ImageNet 预训练权重下载到 `.cache/torch/` 并校验哈希前缀。`meter_type_groups.json` 固定每张图的组及验证折，同组不跨训练/验证；新增照片后需要先更新分组清单，否则训练会拒绝运行。模型和训练记录位于 `models/meter_type_cnn/`。

ONNX 为 opset 12、FP32，固定输入 `images: [1,3,256,256]`，输出 `logits: [1,2]`，类别顺序 `lcd, wheel`。先应用照片方向，整图等比例缩放后居中补灰色 RGB(128,128,128) 到 256×256，RGB /255，再逐通道减 `[0.485,0.456,0.406]`、除 `[0.229,0.224,0.225]`，最后转 NCHW。输出取 argmax，展示分数时另做 softmax。完整预处理也存于 ONNX 元数据。模型约 6.09 MB；尚未在微信小程序或手机上验证。

早期失败的零样本探索保留为 `prepare_meter_type_clip.py` / `classify_meter_clip.py`，不参与默认分类流程，其旧结果位于 `results/meter_type_fp32/`。

## 画质评分测试

```powershell
uv sync --group export --group quality --cache-dir .cache\uv
.\.venv\Scripts\python.exe assess_quality.py --download-only
.\.venv\Scripts\python.exe assess_quality.py
```

使用 `pyiqa` 的 HyperIQA 和 MUSIQ 默认 KonIQ-10k 预训练权重，在同一批 30 个最大面积 LCD 窗口上评分。输入来自原图裁剪，保持比例缩放至短边 256 像素，采用模型默认预处理（HyperIQA 汇总 25 个局部裁剪）。首次准备会下载模型到项目缓存，并按权重文件名公布的 SHA256 前缀验证完整性；实际评分阶段禁用 Hugging Face 在线元数据查询。

另从 3 张原始窗口生成模糊、白色遮挡、右侧截断的对照图。它们与真实照片分开统计；原先未确认参考值的 5 张仅作为待核对组，不作为“不可读”标准答案。

输出在 `results/lcd_quality/`：`quality.json`、`quality.csv`、两种模型各自的 `*_ranked.jpg`（从低分到高分，越靠前模型认为画质越差）、`controls_overview.jpg`，以及实际参与评分的无框 PNG 输入。两种模型的分数尺度不同，不能直接横向比较，也不是读数正确概率。本实验使用 PyTorch CPU，尚未导出 IQA 的 ONNX 或测试小程序性能。

## 数字识别测试

先运行 `main.py` 生成最大面积窗口，再执行：

```powershell
.\.venv\Scripts\python.exe prepare_model.py --kind digits
.\.venv\Scripts\python.exe verify_export.py --kind digits
.\.venv\Scripts\python.exe recognize.py
```

`prepare_model.py --kind digits` 首次下载同一上游项目的 `digits.pt`，校验 Git blob 哈希并导出 `models/digits.onnx`。后续识别只需 `recognize.py`，使用 ONNX Runtime，不加载 PyTorch。

识别使用 `results/lcd_largest/detections.json` 中选中的原图窗口坐标，从原图重新裁剪，避免依赖 JPEG 裁剪文件再次压缩后的像素。数字模型采用类无关 NMS 去除同一位置的重复类别框，忽略 `dot/minus`，按数字框中心从左到右拼接。按约定将最后两位解释为小数，保留原始数字串和前导零。

结果：`results/lcd_recognition/overview.jpg`（数字框和预测/人工参考对照）、`annotated/`、`readings.json`、`readings.csv`。JSON 中包含每个数字的位置、类别、分数以及整条读数的最小/平均数字分数；这些分数未经正确率校准。

`lcd_reference.json` 是助手按图转录并放大复核的临时人工参考，无法确认的图片标记为 `null`，不计入准确率。初次转录的更正保留 `initial_transcription` 与复核原因，便于用户检查，不能当作独立标注测试集。

所有读数均是 `candidate` 或 `review`，不会自动接受。窗口触及原图边缘会标记 `review`；本版尚未实现自动反光/模糊拒识，也没有使用参考标签干预模型预测。

## 环境与运行

在本目录执行 PowerShell 命令。缓存、临时文件和模型均保存在项目内。

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:TEMP = Join-Path $PWD '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force .cache,.tmp | Out-Null
uv sync --group export
.\.venv\Scripts\python.exe prepare_model.py
.\.venv\Scripts\python.exe verify_export.py
.\.venv\Scripts\python.exe main.py
```

已经有 `models/display.onnx` 时，只需运行 `main.py`；运行依赖为 NumPy、OpenCV、ONNX Runtime。`export` 依赖组仅用于下载后的 PyTorch 模型导出和比对，不是小程序依赖。

```powershell
# 自选照片 / 文件夹、阈值和结果目录
.\.venv\Scripts\python.exe main.py --input ..\datas\lcd --conf 0.25 --output results\lcd_largest
# 同批照片上的阈值探索（不是独立测试集）
.\.venv\Scripts\python.exe main.py --conf 0.75 --output results\lcd_largest_conf075
```

不同实验应使用不同的 `--output`，避免旧的裁剪图与新结果混淆。

## 模型

- 来源：[adbartx/multimeter-ocr](https://github.com/adbartx/multimeter-ocr)，`models/display.pt`。
- 原用途：仪表显示窗口检测。这里直接测试它对电表 LCD 的泛化，不把原项目效果视为本项目效果。
- 下载文件：5,502,801 字节；Git blob SHA1：`be6cd6c5646b2d0d0390e88f5871bbaf225bc551`。
- SHA256：`26cac8eebb331fb10d89ab7eb79c391e1954ab784a3e97d971cc12bb5080a74b`。
- `prepare_model.py` 校验上述 Git blob 哈希，并保存来源清单 `models/display.source.json`。若远端文件改变，校验失败，不会静默使用新权重。
- 上游仓库为 AGPL-3.0，权重来源及相关说明见上游仓库。

导出设置：ONNX opset 12、FP32、batch=1、固定输入 `[1,3,640,640]`，不内置 NMS、不做图简化。ONNX 能在电脑加载不代表微信推理引擎已验证兼容。

## 输入输出约定

1. 等比例缩放并补边到 640×640，补边值 114。
2. BGR 转 RGB，NCHW、float32、像素除以 255。
3. ONNX Runtime CPU 推理，默认 4 线程，会话复用并先预热一次。
4. 按模型类别分数筛选，逐类别 NMS，默认分数阈值 0.25、IoU 阈值 0.45。
5. 去除补边并映射回原图，裁剪到原图边界。
6. 在剩余候选中只保留原图面积最大的框；面积相同时取分数更高的框，无候选时不输出框。

结果默认保存在 `results/lcd_largest/`，之前未做面积筛选的结果保留在 `results/lcd/` 和 `results/lcd_conf075/`：

- `overview.jpg`：所有输入的检测框总览。
- `top_windows.jpg`：每张图选出的最大面积窗口裁剪总览，只供检查，不代表自动接受结果。
- `annotated/`：原尺寸检测框图片。
- `crops/`：检测框内的原始图像。
- `detections.json`：选中框的原图坐标 `xyxy`、分数、筛选前候选数量、选择规则、输入尺寸、模型哈希、实际运行库版本和耗时。`best_raw_score` 为模型候选的最高分，不一定等于选中框的分数。

`verify_export.py` 在首张、中间、末张照片上比较 PyTorch 与 ONNX 原始输出，并对照 Ultralytics 官方预测得到的全部 NMS 候选框（面积筛选之前），记录到 `results/export_verification.json`。这用于检查模型导出及前后处理，不能证明框到了正确的 LCD。

**有框的图片数量不是准确率，检测分数也不是读数正确概率。** 当前没有人工真实框，定位效果还需看原图和裁剪；耗时为本机 CPU 耗时，不代表手机。`processing_ms` 包含图片解码、预处理、推理和后处理，排除绘图和结果文件写入。
