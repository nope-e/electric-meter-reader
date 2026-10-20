# 轮位检测模型训练与识别对照

2026-10-20。用户要求用模型替换逐位轮廓切分，并明确电表有 5 位和 6 位。本次新增单类 `wheel_slot` 检测器，目标是一整个轮位窗口，过渡态上下两个半字合为一框；不识别数字值，不将螺丝、小数点、单位标为轮位。

流程为：整图类型分类 → 原 area 模型 → 轮位 YOLO11n → 原 VisionState ONNX。前两阶段及 VisionState 权重、预处理和进位解码不变。轮位数量来自模型经置信度过滤和 NMS 后的实际框数，按框中心从左到右排序；不依据颜色指定 5/6 位，不进行轮位外推或补位。数量异常只提示复核，保留原始检测结果。

## 数据与标注

`datasets/wheel_slots/` 共 73 张 area 裁图、416 个轮位；其中 22 张为五位、51 张为六位。

| 划分 | 图片 | 轮位框 |
| --- | ---: | ---: |
| train | 65 | 371 |
| valid | 5 | 29 |
| test | 3 | 16 |

- 公开来源为 `Electricity meter.v5i.yolov8` 的 22 张滚轮图，许可声明 CC BY 4.0。沿用原 train/valid/test 划分（14/5/3）。原标签是数字字形框，部分红字与过渡轮位漏标；以字形中心提出轮位网格，再由助手目视检查并补齐，转换为完整轮位窗口框。
- 本地 `datas/wheel` 有 53 张，51 张得到完整、可核对的 area 后用于训练。1 张上游没有 area、1 张 area 截断末位，不将漏标轮位当背景。原规则提供初框，助手检查七页联系表，补标五张旧规则切分失败的蓝色六位图。
- 本地两组重复照片整组都在 train，不随机抽几张同表照片做验证。73 张不代表 73 台独立电表；公开不同原图名也不能保证不同实体。
- 第一、二版未用 `datas/test_wheel` 训练或验证选权重。第三版按用户授权抽取部分照片加入训练，具体分组见下文；这批照片属于持续使用的开发集。

标注审核记录为 `wheel_slot_reviews.json`，绑定原图 SHA256。`manifest.json` 保存来源、组、裁剪坐标、框、源图/输出图/标签哈希；生成与训练前核验输出清单，检查同源组及相同裁图像素不跨集合。标注为助手初核，尚无独立人工验收。

### 第二版：背景与负样本

首版训练裁图太紧，导致真实两侧螺丝、圆形标记和单位文字缺少监督：虽然 3 张公开 test 的 mAP50 为 .995，但在 18 张开发对照图的 14 张有 area 图片上，全部出现额外误框（实际输出 6～8 框，合计 99 框）。该高分不能代表当前场景可用。首版权重与失败预览分别保留在 `slots_yolo11n_v1/`、`results/test_wheel_slot_model/`，不掩盖这次失败。

用户进一步要求加入负样本，第二版 `datasets/wheel_slots_negatives/` 包含：

- 73 张原始正样本裁图扩大上下左右范围，轮位正框不变，更多商标、圆圈标记、单位、金属件成为真实背景。
- 从原训练照片中人工框选 6 张只含螺丝、圆形封帽/端子的背景裁图，YOLO 标签文件为空。
- 在 37 张训练读数区左右加入来自这 6 个真实背景块的金属件合成图，包含 5/6 位、不同大小与亮度，保持真实轮位标签不变。合成图只入 train，不能算新增独立照片。

共 116 张导出图、619 个正框：train 108 张/574 框（含 6 张纯负样本及 37 张合成图），valid 5 张/29 框，test 3 张/16 框。审核记录为 `wheel_slot_negative_reviews.json`，合成来源及哈希保存在 `augmentation_sources`；所有素材来自原训练侧，不使用 `datas/test_wheel`。四页合成图与真实背景块已目视检查。

### 第三版：完整窗口框与实拍适配

用户指出第二版裁图过紧，并授权把部分测试照片加入训练。第三版 `datasets/wheel_slots_local/` 保留第二版数据，将正框宽度扩大 18%、高度扩大 12%，标整段窗口而非仅数字笔画；新增 14 张有 area 的实拍正样本和 12 张实拍背景负样本。12 张负样本只从新增训练组中挖取，均不与轮位标签相交，目视检查为螺丝或商标。

同一块表的多张照片整组划分，避免把同一台表的近似照片分到训练和检查两侧：

| 新增照片组（文件名末四位） | 用途 | 张数 |
| --- | --- | ---: |
| 9494 / 9554 / 9632；9923 / 9989；0036 / 0134 | train，3 组 | 7 |
| 0198；0444 | valid，2 组 | 2 |
| 0319 / 0578；0639 / 0696 / 0762 | test，2 组 | 5 |
| 9756 / 9825 / 0073 / 0163 | 上游未检测到 area，未加入轮位数据 | 4 |

第三版总计 142 张导出图、689 个正框；train 127 张/609 框，valid 7 张/39 框，test 8 张/41 框。包含 55 张五位、69 张六位、18 张纯背景；合成图和重复照片不算独立电表。标注记录 `wheel_slot_local_reviews.json`、数据清单及摘要均记录授权用途和分组，原测试目录不移动、不改图。

识别裁图另在检测框每侧增加 8% 宽、10% 高的留边，从原照片读取像素，因此可越过 area 边界取回轮位边缘。原始检测框和实际裁图坐标分别保存，预览绿框为检测框、橙框为最终裁图范围。上游分类、area 权重以及 VisionState 权重、预处理、解码均未改变。留边不等同于补出不存在的检测位。

轮位检测输入默认 `--area-context .10`：area 左右各扩原宽的 10%，上下各扩原高的 11.67%。仅用新增 7 张 train 和 2 张 valid 比较 0/.05/.10/.15：0 时 valid 漏一位，.05 时 train 多一框，.10/.15 时这 9 张均完整匹配 5 位，因此采用较小的 .10，随后才对留出照片运行该设置。记录在最终结果目录的 `context_calibration.json`。该参数只调整检测输入，area 模型本身不变；可用 `--area-context 0` 复现紧贴 area 的第三版对照。

## 训练与导出

脚本 `train_wheel_slots.py` 首版复用现有 `models/meter_area/area.pt` 微调 YOLO11n，单类名称改为 `wheel_slot`。最大 160 轮、patience=35，实际早停 119 轮，选择第 84 轮。第二版从首版 best.pt 继续微调，最多 120 轮、patience=30，实际 65 轮、best=35。第三版从第二版 best.pt 微调，最多 120 轮、patience=30，实际 57 轮、best=27，训练、评估及导出共约 232 秒。AdamW lr0=.001、cos_lr、seed=20261020、640 输入、batch=12、单 GPU、workers=0、AMP 关闭；无水平/垂直翻转，保留轻微旋转、透视、亮度与 mosaic 增强。完整参数、逐轮状态和 CSV 在 `models/wheel_slots/`。

导出目标为 `models/wheel_slots/wheel_slots.onnx`：FP32、opset 12、输入 `[1,3,640,640]`，输出 `[1,5,8400]`，无内置 NMS。RGB /255、NCHW、保持比例缩放并居中补 114；参考实现为 `main.preprocess/decode`。默认 conf=.25、NMS IoU=.45。

运行时只需要 ONNX Runtime、NumPy、OpenCV、Pillow，不加载 PyTorch。最后一轮按既定规则向下取整；默认小数格式一位由 `--decimal-places 1` 明确设置，检测器不推断小数位置。始终保存原始数字串与前导零。

训练/导出方式参照 [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/) 和 [导出文档](https://docs.ultralytics.com/modes/export/)，实际使用本地已安装版本和已有权重，无新增在线权重下载。

## 复现

在 `test-prj` 执行，已有输出时使用新目录/运行名：

```powershell
# 已审核标注的可复现构建；原下载数据不修改
.\.venv\Scripts\python.exe prepare_wheel_slot_data.py --output datasets\wheel_slots_rebuilt
# 根据已审核的负样本记录重建第二版数据（需要保留合成素材目录）
.\.venv\Scripts\python.exe prepare_wheel_slot_data.py --review wheel_slot_negative_reviews.json --output datasets\wheel_slots_negatives_rebuilt
# 训练、公开 test 评估、ONNX 导出与 PT/ORT 一致性核验
.\.venv\Scripts\python.exe train_wheel_slots.py --data datasets\wheel_slots_negatives --weights models\wheel_slots\slots_yolo11n_v1\weights\best.pt --name slots_yolo11n_recheck --epochs 120 --patience 30
# 第三版：使用已审核的授权实拍适配标注
.\.venv\Scripts\python.exe prepare_wheel_slot_data.py --review wheel_slot_local_reviews.json --output datasets\wheel_slots_local_rebuilt
.\.venv\Scripts\python.exe train_wheel_slots.py --data datasets\wheel_slots_local --weights models\wheel_slots\slots_yolo11n_v2_negatives\weights\best.pt --name slots_yolo11n_v3_recheck --epochs 120 --patience 30
# 相同冻结分类/area 的对照
.\.venv\Scripts\python.exe recognize_wheel_slots.py --windows results\test_wheel_area_v2\detections.json --output results\slot_model_recheck
# 新图完整流程：整图分类、area、轮位、VisionState
.\.venv\Scripts\python.exe recognize_wheel_slots.py --input ..\datas\test_wheel --output results\slot_model_fresh
.\.venv\Scripts\python.exe -m unittest test_wheel_slot_detection test_visionstate_wheel test_wheel_decoding test_area_selection test_merge_window_data -v
```

24 项单元回归通过，新增检查包括 5/6 位变长排序、缺位不补框、边缘裁剪像素、留边取回框外像素和小数格式保留前导零。模型定位效果需要另看真实数据与逐图预览，单元测试不代表定位准确率。

## 结果

当前权重为第三版 `models/wheel_slots/wheel_slots.onnx`，10,564,981 字节，SHA256 `101fbd4de98290650fd0a5f8d90abf9b57b7982a8fc20e2a49de04367e2b17ac`。原始 best.pt/ONNX 保留在 `models/wheel_slots/slots_yolo11n_v3_local/weights/`，前两版也保留。

第三版混合 test（3 张公开 + 5 张实拍，41 个框）mAP50=.993636、mAP50-95=.616536。部署 conf=.25 时，数据集导出裁图上 train 609/609 框匹配、valid 37/39、test 39/41（IoU≥.5）；test 输出 40 框，含一额外框。AP 与最终读数不同，公开裁图、扩展后的标注裁图和实际 area 输入也不同。

### 18 张实拍完整流程

最终目录 `results/test_wheel_slot_model_v3_fresh/`，从原照片重新运行分类、area、轮位与 VisionState，全部使用 ONNX Runtime CPU。18 张的状态、框数、候选与同设置的冻结 area 回放一致。目视检查两页检测框及两页实际 32×48 模型输入，留边后正常数字及过渡态轮位完整度明显改善。

| 分组 | 张数 | 完整匹配轮位且无额外框 | 读数与助手目视参考一致 |
| --- | ---: | ---: | ---: |
| 新增 train | 7 | 7 | 7 |
| 新增 valid | 2 | 2 | 2 |
| 新增 test，未参与训练/选 context | 5 | 4 | 4 |
| 上游无 area，未分配轮位用途 | 4 | — | 无候选 |

“完整匹配”要求每个已审核轮位均在 IoU≥.5 下匹配且没有多余预测，不只是框数碰巧等于 5。读数参考为助手目视初核、沿用未完成进位取前一位的约定，尚未经过用户逐图确认；这批开发图此前已做过多次实验，不是新的盲测。

有 area 的 14 张中共 13 张与目视参考一致（第二版为 4 张）。其中 7 张已训练、2 张用于验证，因此不将 13/14 当泛化准确率。当前 14 张均未见螺丝框，但 `1792476680578.jpg` 在五个真轮位之外仍多框右侧外壳纹理：分数 .313，输出 6 位 `76172.9`，目视参考 `7617.2`。没有按“应有五位”删掉这个多框来掩盖问题。

| 文件末四位 | 用途 | 最终候选 |
| --- | --- | --- |
| 9494 / 9554 / 9632 | train | `0179.2` |
| 9923 / 9989 | train | `2208.9` |
| 0036 / 0134 | train | `0707.1` |
| 0198 | valid | `7892.1` |
| 0444 | valid | `5369.8` |
| 0319 | test | `7617.2` |
| 0578 | test | `76172.9`，额外背景框 |
| 0639 / 0696 / 0762 | test | `9498.4` |
| 9756 / 9825 / 0073 / 0163 | 上游无 area | 无候选 |

复核入口：`overview_00.jpg` / `overview_01.jpg` 为框和候选；`slots_00.jpg` / `slots_01.jpg` 为实际模型输入；`readings.json` 保存框、裁图坐标、分数、状态与哈希；`adaptation_evaluation.json` 保存逐图分组、目视参考和 IoU 匹配。

### 五位、六位与运行核验

原 86 张回放在 `results/area_wheel_slot_model_v3_context/`：53 张 wheel、52 张有 area，52 张均输出轮位；在有完整轮位标注的原图中，五位表 14/14 输出五框，六位表 34/37 输出六框。它们主要参与过训练，且框数正确不等于读数正确；仍可见模糊图漏末位和识别错误，不能宣称六位全部解决。

24 项单元回归通过；ONNX checker 通过；9 张 train/valid/test 裁图的 PT/ORT 原始输出及后处理框一致性通过，最大原始输出绝对误差 .001511。VisionState SHA256 仍为 `37912f885b061a7b12298872fbdfa1b73ca0664363b293a55617b1d3680a98ac`。完整 ONNX 流程已在本机运行，微信小程序运行库兼容性和手机性能尚未验证。
