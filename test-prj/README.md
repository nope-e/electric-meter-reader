# 电表类型分类、LCD 窗口检测与数字识别实验

当前部署模型和完整输入输出/前后处理说明已整理到项目根目录 [models/README.md](../models/README.md)，包含 5 个 ONNX、文件哈希、独立 ONNX Runtime 参考流程和 104 张本地照片的一致性核验。原实验模型目录继续保留。

LCD 窗口定位和数字识别使用现成模型；新增的 LCD / 滚轮表分类使用本地照片微调 MobileNetV3-small。LCD 原图输入默认为 `../datas/lcd`，类型分类默认读取 `../datas`。

全部 104 张自拍照片的最新统一回测见 [ALL_LOCAL_RESULTS.md](ALL_LOCAL_RESULTS.md)：98 张有可评分参考，整串一致 79 张（80.6%）；LCD 27/28（96.4%），wheel 52/70（74.3%）。另 6 张参考不确定单列；漏窗口计失败。可用 `evaluate_local_meters.py --output results\all_local_meters_recheck` 重跑，逐图结果在 `results/all_local_meters/evaluation.csv`。

## 滚轮表：模型检测轮位 + VisionState 识别

新增入口 `recognize_wheel_slots.py`：整图分类 → area → 单类 YOLO11n 轮位检测 → VisionState ONNX。轮位数由实际检测决定，支持 5/6 位，不进行固定数量补位。轮位权重为 `models/wheel_slots/wheel_slots.onnx`，VisionState 继续使用 `models/visionstate/wheels-v1.onnx`；原 LCD 识别模型不变。

```powershell
# 新图完整流程；输出目录需要不存在
.\.venv\Scripts\python.exe recognize_wheel_slots.py --input ..\datas\test_wheel --output results\wheel_fresh
# 同一分类/area 结果下比较轮位模型
.\.venv\Scripts\python.exe recognize_wheel_slots.py --windows results\test_wheel_area_v2\detections.json --output results\wheel_recheck
```

第三版按用户授权加入 7 张实拍训练照片，同一块表的照片整组划分，另有 2 张验证、5 张分组留出；原 18 张目录已用于适配，不能再把整批回测当独立测试准确率。标注扩大为完整轮位窗口，默认给 area 读入少量周边像素（`--area-context .10`）；轮位裁图每侧留宽 8%、高 10%，从原照片取像素。实际框、模型输入和读数对照见 [WHEEL_SLOT_RESULTS.md](WHEEL_SLOT_RESULTS.md)。旧 OpenCV 切位及 class100/VisionState 固定裁图比较保留在原入口和历史报告中。

## 整图分类 + area 窗口检测

当前实验采用两个独立 ONNX 模型：整图 MobileNet 判断 `digital/wheel`，YOLO11n 只检测一个 `area` 类。整图分类复用 `models/meter_type_cnn/meter_type.onnx`，在组合接口把原 `lcd` 类名映射成 `digital`；分类依据完整图片像素，读数区检测不会决定表型。area 训练及 ONNX 核验已完成，改用置信度优先选框后，本地目视读数区位置覆盖为 digital 33/33、wheel 52/53（包括部分裁边情况，不代表完整数字覆盖），详见下方报告。

单类检测数据来自已复核的 601 张图、609 个窗口，只把原来的 lcd/wheel 框合并成 area，图片、坐标及 train/valid/test 划分保持一致。未将整张图框入 area。

```powershell
# 仅首次准备数据；已有输出时会拒绝覆盖
.\.venv\Scripts\python.exe prepare_area_data.py
# 单类 area 对照训练，自动测试、导出、核验
.\.venv\Scripts\python.exe -u train_meter_windows.py --single-area
# 训练完成后，组合两个模型测试本地 lcd 与 wheel 文件夹
.\.venv\Scripts\python.exe detect_meter_area.py
# 任意文件或文件夹，输出使用新目录
.\.venv\Scripts\python.exe detect_meter_area.py --input photo.jpg --output results\area_single
```

area 模型输出为 `models/meter_area/area.onnx`，训练状态为 `models/meter_area/area_yolo11n.status.json`。组合接口先做 conf=0.25 过滤和 IoU=0.45 的 NMS，再选置信度最高的候选，仅同分时取面积较大的框。保存所有候选坐标供检查；预览灰框表示其他候选，绿框表示选中的 area。默认输出 `results/meter_area_confidence/detections.json`、原图预览、裁剪及分页面总览。重复运行需换 `--output`，避免覆盖既有对照结果。可用 `--selection largest_area` 复现旧选法，默认是 `--selection confidence`。
对照设置、评估口径和结果见 [AREA_RESULTS.md](AREA_RESULTS.md)。

Python 接口可复用会话：

```python
from PIL import Image, ImageOps
from detect_meter_area import MeterAreaPipeline

pipeline = MeterAreaPipeline()
with Image.open("photo.jpg") as photo:
    result = pipeline.predict(ImageOps.exif_transpose(photo).convert("RGB"))
print(result["meter_type"], result["detections"])
```

使用当前分类与 area 结果，只对预测为 digital 的图片运行原数字识别模型：

```powershell
.\.venv\Scripts\python.exe recognize.py --windows results\meter_area_confidence\detections.json --meter-type digital --output results\area_digital_recognition
```

已测试 33 张 digital，原有人工参考 25 张中 24 张完全一致（96%）；`1792403668125.jpg` 漏掉末位 6，其余 8 张没有可评分参考。读数模型和识别阈值未改变，详细结果与旧窗口对照见 [RECOGNITION_RESULTS.md](RECOGNITION_RESULTS.md)。

整图分类模型用全部 86 张本地照片训练过，原图分类匹配属于训练集回测。area 对照实验不使用这些原图训练。旧两类窗口模型仅忽略类别的几何基线位于 `results/meter_area_previous/`，复现命令为：

```powershell
.\.venv\Scripts\python.exe detect_meter_area.py --area-model models\meter_windows\meter_windows.onnx --collapse-types --selection largest_area --output results\area_previous_recheck
```

实测结论、误检案例及运行边界见 [TEST_RESULTS.md](TEST_RESULTS.md)。
数字识别结果及人工对照说明见 [RECOGNITION_RESULTS.md](RECOGNITION_RESULTS.md)。
LCD 画质/可读性探索见 [QUALITY_RESULTS.md](QUALITY_RESULTS.md)。
类型分类的全量训练结果与历史分组验证见 [METER_TYPE_RESULTS.md](METER_TYPE_RESULTS.md)。
两类读数窗口检测的数据合并规则、数量与核验结果见 [WINDOW_DATA_RESULTS.md](WINDOW_DATA_RESULTS.md)。新模型已完成训练与 ONNX 导出，测试集及本地原图结果见 [WINDOW_TRAINING_RESULTS.md](WINDOW_TRAINING_RESULTS.md)。本地滚轮表仍有大量漏检、误分类和错框。

## 两类读数窗口数据

历史 [VisionState ONNX 滚轮识别器对照](VISIONSTATE_RESULTS.md)：保持旧切位不变，在同批 13 张有候选图片中，助手目视初核从 class100 的 2 张提升至约 10 张，待用户确认。新轮位模型入口见上文。

滚轮测试图及删图后的旧切位复测见 [TEST_WHEEL_RESULTS.md](TEST_WHEEL_RESULTS.md)：18 张，分类 18/18、area 14/18、旧切位推理 13/18；这些是新轮位检测前的历史基线。

已把 `../datas` 下四个 Roboflow YOLO 导出合并到 `datasets/meter_windows/`，训练配置为 `datasets/meter_windows/data.yaml`。类别顺序是 `lcd, wheel`：`lcd` 包含 LCD、LED 等电子数字屏，`wheel` 为机械数字滚轮。模型目标是整条读数区域。

`meter-reader-detection` 保留 `Reading` 框的位置，根据对应的 `Digital/Analog` 整表标签赋予 `lcd/wheel` 类别；整表框和序列号框不参与训练。原始下载内容保持不变。

```powershell
uv sync --group data --cache-dir .cache\uv
# 核验已生成的数据
.\.venv\Scripts\python.exe merge_window_data.py --verify-only
# 复现合并，使用新目录，避免覆盖已有结果
.\.venv\Scripts\python.exe merge_window_data.py --output datasets\meter_windows_rebuilt
.\.venv\Scripts\python.exe -m unittest test_merge_window_data -v
```

训练环境的 `export` 依赖组使用 CUDA 12.8 PyTorch，已在本机 RTX 4060 Laptop GPU 上验证 CUDA 计算：

```powershell
uv sync --group export --group data --cache-dir .cache\uv
.\.venv\Scripts\python.exe -u train_meter_windows.py
# 中断后恢复同一次训练
.\.venv\Scripts\python.exe -u train_meter_windows.py --resume
```

默认以 `models/display.pt` 的 YOLO11n 窗口检测权重微调两类，输入 640、batch 16、最多 100 轮、验证集 mAP50-95 连续 20 轮无改善时早停。使用 499 张训练图片，60 张验证图片选最优权重，最后只评估一次 42 张测试图片。训练使用 FP32，禁用新增的水平/垂直翻转。

训练记录和权重位于 `models/meter_windows/reading_yolo11n/`；当前启动命令的日志为 `results/window_training.log`。每轮更新 `models/meter_windows/reading_yolo11n.status.json`。脚本完成后自动评估测试集、导出固定 `[1,3,640,640]`、FP32、opset 12 ONNX，并在六张测试图片上比较 PyTorch/ONNX 原始输出与 NMS 后的框及类别。核验通过后才生成 `models/meter_windows/meter_windows.onnx`，原有 `display.onnx` 保留。

已用现有运行器测试本地原图，仍采用最大面积窗口；结果与预览位于下列命令对应的输出目录：

```powershell
.\.venv\Scripts\python.exe main.py --model models\meter_windows\meter_windows.onnx --input ..\datas\wheel --output results\window_wheel
.\.venv\Scripts\python.exe main.py --model models\meter_windows\meter_windows.onnx --input ..\datas\lcd --output results\window_lcd
```

## LCD / 滚轮表分类

当前 wheel 分类与 area 结果已接入对话中提到的 AI-on-the-edge 单轮位 `dig-class100 / dig-cont` 预训练模型：

```powershell
uv sync --cache-dir .cache\uv
.\.venv\Scripts\python.exe prepare_wheel_model.py
# 首次转换见 WHEEL_RECOGNITION_RESULTS.md；已有 models/wheel/*.onnx 时直接运行
.\.venv\Scripts\python.exe recognize_wheel.py
```

53 张预测 wheel 中，52 张有 area，46 张完成切位并运行识别。清晰图能得到 `00373.3`、`3661.3` 等候选，斜拍/模糊、切位误差和整数附近状态取整仍不稳定，未统计独立准确率。两个滚轮模型均已转为 ONNX，默认使用 ONNX Runtime；262 个实际轮位和 434 个合成输入的两个模型输出均与原生 TFLite 逐值一致。当前结果在 `results/area_wheel_onnx/`，原生基线保留在 `results/area_wheel_readings/`。输入为 float32 RGB NHWC `[1,32,20,3]`、像素 0～255，不做 /255。转换、核验和小程序验证边界见 [WHEEL_RECOGNITION_RESULTS.md](WHEEL_RECOGNITION_RESULTS.md)。

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
