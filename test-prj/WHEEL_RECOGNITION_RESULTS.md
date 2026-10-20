# 现成滚轮模型接入测试

本文为早期 class100/cont 接入记录。当前新增入口使用轮位检测模型与 VisionState ONNX，训练、完整窗口裁图和实拍分组结果见 [WHEEL_SLOT_RESULTS.md](WHEEL_SLOT_RESULTS.md)；下方历史结果保留。

日期：2026-10-19。已读取根目录 `conversation-电表读数实时检测与置信度判断-20261019-175911.json`，定位到消息 18396/18398：原先提到的是 [AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device) 的单轮位 `dig-class100` 和 `dig-cont` 模型；半字约定为输出进位前数字。对话另有 Fast-OCR 备选，本次未使用。

本次接入两个官方原始预训练权重，没有训练或修改权重：

| 模型 | 文件 | 大小 | SHA256 |
| --- | --- | ---: | --- |
| class100 | dig-class100-0182-s2_q.tflite | 226,656 字节 | 78f8ee04f8e195ef9e8e1314c77690d530ef151e24c1cf7ab3c66055d0c2b950 |
| cont | dig-cont_0900_s3_q.tflite | 315,544 字节 | 8f719c03e69d077809ac8bb20d2f79e73172eac652b18dc8d021952b68262a58 |

固定上游 commit `cdc8450924e9338ab564c3fd76ad998e7b7cbb84`，下载后核验 Git blob SHA1。来源与授权文件链接保存在 `models/wheel/source.json`；原始文件位于 `models/wheel/`。

## 接入方式

复用当前 `results/meter_area_confidence/detections.json`：整图分类得到 wheel 后，使用置信度优先选中的 area；不按目录标签筛选，不重新分类或重跑 area。数字屏读数流程保持原样。

两个模型都要求单个轮位输入，不能直接输入整条数字。当前用图片颜色区分本地蓝色六位/绿色五位布局，再用白色字形轮廓寻找等间距、近似共线的黑底数字行，估计轮位中心、倾斜方向和高度，并外推末位红色轮位。它是针对两种固定布局的图像处理基线，没有训练逐位检测器，没有按文件名配置坐标，也没有填入已知读数。局部仿射方向裁剪只处理行内倾斜，并未估计四角或消除完整透视。

每位裁剪缩放至 20×32，输入 float32 RGB NHWC `[1,32,20,3]`，**直接使用 0～255 像素值，不做 /255**。这是按固定版本 [CTfLiteClass.cpp 的 LoadInputImageBasis](https://github.com/jomjol/AI-on-the-edge-device/blob/cdc8450924e9338ab564c3fd76ad998e7b7cbb84/code/components/jomjol_tfliteclass/CTfLiteClass.cpp) 核对的行为，不能套用 YOLO 预处理。

class100 的 argmax 类别 /10 为轮位状态，整数部分为本次候选数字。cont 采用官方源码中主峰与相邻输出的加权插值，处理 9→0 环绕，再取整数部分。采用对话中的“未完成进位取前一数字”约定，未接入官方跨位纠错/历史值算法。原始状态和模型输出保存供复核；直接取整对整数附近的预测误差很敏感，例如 3 的状态略小于 3 会输出 2，0 的状态略小于 10 会输出 9。最后一位按这两种本地表的红色 0.1 位显示小数，保留原始串和前导零，所有输出均需复核。

## 实测结果

| 项目 | 数量 |
| --- | ---: |
| 原始分类结果 | 86 |
| 预测为 wheel，进入分支 | 53 |
| 预测为 digital，跳过 | 33 |
| wheel 中有 area | 52 |
| 完成切位并运行两个识别模型 | 46 |
| 有 area 但切位失败 | 6 |
| area 漏检 | 1 |
| 两模型完整候选串相同 | 22/46 |

46 只是完成切位/推理的数量，**不是正确读数数量**；模型相同输出也不等于正确。尚无独立滚轮读数参考，不报准确率。助手已查看最终全部五页窗口与切位框预览，以及典型单轮位输入，发现切位高度/边缘与识别/取整错误同时存在。

有代表性的输出：

- `1792416423981.jpg`：两模型都输出 `003733`，候选 `00373.3`，目视相符。
- `1792416424962.jpg`：两模型都输出 `36613`，候选 `3661.3`；红位在 3→4 过渡中，符合本次半字约定。
- `1792403667565.jpg`：class100 输出 `35520`，候选 `3552.0`，红位在 0→1 过渡中；cont 首位取整得到 2，输出 `25520`。
- `1792416424039.jpg`：数字可见为 `003733`，但 class100 输出 `992633`、cont 输出 `993732`，存在整数边界取整和输入适配问题。
- `1792416425142.jpg`：黑位可见为 `3661`，两模型却都输出 `76603`，说明不能以一致性作为正确证明。
- `1792416424834.jpg`：模糊导致白色字形仅剩部分高亮，切位高度不足，两个输出都不正确。

无 area：`1792416423878.jpg`。切位失败：`1792403667969.jpg`、`1792416424791.jpg`、`1792416424803.jpg`、`1792416424819.jpg`、`1792416424849.jpg`、`1792416425011.jpg`。最后一张本身 area 已裁掉红色小数位。完整记录在 `results/area_wheel_readings/visual_review.json`。

结论是单轮位权重能识别部分清晰样本和过渡态，但当前整链路不稳定；后续需要更可靠的轮位定位、完整透视校正，并验证整数附近的状态解码，不能只把模型换进去就当作可用读数器。

## 运行、验证与产物

```powershell
uv sync --cache-dir .cache\uv
.\.venv\Scripts\python.exe prepare_wheel_model.py
# 已有两个 ONNX 模型时：
.\.venv\Scripts\python.exe recognize_wheel.py
# 已有输出时，使用新目录
.\.venv\Scripts\python.exe recognize_wheel.py --output results\wheel_recheck
.\.venv\Scripts\python.exe -m unittest test_wheel_decoding test_area_selection test_merge_window_data -v
```

最终目录 `results/area_wheel_readings/` 包含 `readings.json`、`readings.csv`、五页 `overview_00.jpg` 至 `overview_04.jpg`、`slots_*.jpg` 单位预览、`slots/` 原始轮位裁剪和 `inputs/` 模型实际 20×32 输入 PNG。JSON 保存每位四角、两模型状态及输出；无检测和切位失败也保留记录。早期切位探索目录 `area_wheel_recognition/`、`area_wheel_recognition_v2/` 保留作历史，最终以 `area_wheel_readings/` 为准。切位规则在这批图片上调整过，尚无独立验证。

分流集合与原预测 wheel 集合逐项核验一致，权重哈希与下载清单一致。14 项单元回归检查通过，含 3→4、9→0 状态解码边界。CPU 单轮位纯推理中位数 class100 约 0.096 ms、cont 约 0.192 ms，不含切位、窗口检测、解码或文件输出，不代表手机表现。

此前使用 LiteRT 2.3.0 建立原生 TFLite 基线。tflite2onnx 0.4.1 曾因不支持逐通道量化而失败；2026-10-20 已通过下面的方案完成 ONNX 转换，默认运行库改为 ONNX Runtime。

## ONNX 导出与核验（2026-10-20）

| 模型文件（`models/wheel/`） | 字节数 | ONNX SHA256 |
| --- | ---: | --- |
| `dig-class100-0182-s2_q.onnx` | 238,105 | `d0f07f6542d2fa7685da839a4563fa3e387d2a3d59357fa44ba345fb54fcc777` |
| `dig-cont_0900_s3_q.onnx` | 330,561 | `bc7f6040967dce067ac00fd9d297b61cfa6793e76986d754f1e2eb6bd017159e` |

tf2onnx 1.17.0 / TensorFlow 2.20.0 导出，ONNX opset 13、IR 7，固定输入 float32 RGB NHWC `[1,32,20,3]`，0～255 不做归一化。输出分别为 `[1,100]` 与 `[1,10]`。保留原量化参数和权重，未训练或校准。

直接转换 cont 时有 5/262 个轮位最终数字变化，原因是量化 LeakyReLU 的整数舍入语义不同。导出脚本对每个原始 LeakyReLU 算子的全部 256 个 int8 输入运行原算子，生成精确查表，以 ONNX 标准 Cast/Add/Gather 实现，共 5 张表。不使用测试图片拟合查表，不包含自定义算子。

`verify_wheel_onnx.py` 在 ONNX Runtime 1.31.0 CPU 单线程与 LiteRT 2.3.0 单线程之间核验：

| 输入 | 每个模型的数量 | class100 原始输出完全相同 | cont 原始输出完全相同 |
| --- | ---: | ---: | ---: |
| 原生基线保存的实际轮位 PNG | 262 | 262/262 | 262/262 |
| 合成输入：256 级纯色、128 个固定种子随机 RGB、50 个横竖阶跃边缘 | 434 | 434/434 | 434/434 |

最大绝对误差均为 0，argmax 和最终取整数字均一致。模型哈希、实际输入 PNG 哈希、形状、算子、版本和逐组结果保存在 `models/wheel/onnx_verification.json`。有限输入对比不能证明所有输入或所有设备均逐值一致，也不是识别准确率评估。

默认 `recognize_wheel.py` 已使用 ONNX Runtime，无需 TensorFlow 或 LiteRT。重跑结果保存在 `results/area_wheel_onnx/`：53 张 wheel 的状态、切位、原始模型输出和候选读数与 TFLite 基线逐项一致，仍是 46 张完成、6 张切位失败、1 张 area 漏检、22 张两模型候选串一致。转换保持既有行为，也保留上述识别错误。ONNX CPU 单轮位推理中位数 class100 约 0.327 ms、cont 约 1.076 ms；不含切位和文件输出，与原生结果不是严格基准测试，未在微信小程序或手机验证算子兼容和性能。

复现转换使用独立环境，避免替换主项目依赖：

```powershell
New-Item -ItemType Directory -Force .tmp,.cache | Out-Null
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
uv venv .venv-wheel-export --python .venv\Scripts\python.exe --cache-dir .cache\uv
uv pip install --python .venv-wheel-export\Scripts\python.exe -r requirements-wheel-export.txt --cache-dir .cache\uv
.\.venv-wheel-export\Scripts\python.exe export_wheel_onnx.py
# 仅核验/原生对照需要这些可选依赖，正常 ONNX 推理无需安装
uv sync --group export --group wheel --cache-dir .cache\uv
.\.venv\Scripts\python.exe verify_wheel_onnx.py
.\.venv\Scripts\python.exe recognize_wheel.py --output results\wheel_onnx_recheck
# 把新整链路结果也与原生基线逐项比较
.\.venv\Scripts\python.exe verify_wheel_onnx.py --readings results\wheel_onnx_recheck\readings.json
# 可选原生对照
.\.venv\Scripts\python.exe recognize_wheel.py --runtime tflite --output results\wheel_tflite_recheck
```

## 新增测试集（2026-10-20）

用户提供 `datas/test_wheel` 的 23 张新图后，固定模型与全部规则运行：23 张均分类为 wheel，18 张有 area，17 张完成切位，两个模型候选串仅 1 张一致。新图暴露出螺丝误入首位、漏末位以及识别/取整错误。85 个实际轮位的两个 ONNX 模型输出仍与原生 TFLite 逐值一致。详细逐图例子、结果目录和评估边界见 [TEST_WHEEL_RESULTS.md](TEST_WHEEL_RESULTS.md)。
