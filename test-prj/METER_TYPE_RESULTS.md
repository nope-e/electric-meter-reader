# LCD / 滚轮表分类测试

使用 ImageNet 预训练 MobileNetV3-small 替换末层为两类，微调后导出 ONNX；运行时仅使用 ONNX Runtime CPU。

## 当前使用：全量训练

用户确认只需覆盖现有表型，因此最终使用全部 **86 张图**训练：LCD 33 张、wheel 53 张，两种滚轮表型全部参与，无照片被留出。`classify_meter.py` 默认加载这个最终模型 `models/meter_type_cnn/meter_type.onnx`。

最终 ONNX 原图回测 **86/86**（LCD 33/33、wheel 53/53），属于训练集回测。结果见 `results/meter_type/overview.jpg` 和 `classifications.json`。此前的 11 张留出错例在这个全量模型下均已分类正确。

后续检查优先使用现有表型的新拍照片，确认不同角度、距离和光照下的效果。当前不以识别未知表型为目标。

## 历史探索：按表型分组验证

用户整理后的目录包含 LCD 33 张、wheel 53 张，3 张新增 LCD 已移动至正确目录。所有图像只通过像素参与模型推理，目录标签仅用于训练或预测后的评估。

新增 wheel 多为重复拍摄，按照片随机拆分会让同一只表进入训练和验证。因此在训练前固定 `meter_type_groups.json`，做两折分组验证：

- 银色蓝条滚轮表 506 的 38 张（含原始 `1792403667969.jpg`）整体留出，另一种滚轮表参与训练。
- 黑色绿条滚轮表的 15 张（含原始 `1792403667565.jpg`）整体留出，另一种滚轮表参与训练。该组保守地合并相似外观，即使可能是不同实物。
- LCD 重复视图 `7284/7313`、`4886/4903` 分别放在同组，其他按目视区分的表分组。该分组为助手人工整理，未经独立审计。

训练启动时检查清单与照片完全一致、组不跨折、文件 SHA256 不跨折、每折训练和验证均包含两类。所有增强只用于训练图。固定种子 20261019，每次 20 epoch，每轮均衡采样 64 张，batch=8；AdamW，特征层 lr=1e-4、分类头 lr=1e-3，冻结 BatchNorm 统计；旋转/平移/缩放、颜色扰动和随机灰度增强。epoch 数在验证前固定，未按验证结果选 checkpoint。

| 验证类别/组 | 正确 / 总数 |
| --- | ---: |
| LCD | 33 / 33 |
| 银色蓝条 wheel | 28 / 38 |
| 黑色绿条 wheel | 14 / 15 |
| wheel 合计 | 42 / 53 |
| 总计 | **75 / 86（87.2%）** |

以上为对应留出折 ONNX 模型的预测。11 张错误均为 wheel 被判为 LCD，包含原来两张 wheel；相似外壳、拍摄角度与反光下的跨表型泛化仍不足。具体样本和实际预处理输入均保留，不能把“当前目录都能分对”理解为对任意新电表都可靠。

上述留出实验用于探索未参与训练的表型，不是默认使用的最终模型。用户当前要求覆盖固定表型，以前述全量训练模型为准；`fold0.onnx`、`fold1.onnx` 仅保留实验记录。

## 模型与运行

- 模型：`models/meter_type_cnn/meter_type.onnx`，6,094,315 字节（约 5.81 MiB）。
- SHA256：`a3e1469e667db1d1fc2d1bd3b1192d952a5a9db896dc88d145d6e5406c6623ea`。
- 输入 `images: float32 [1,3,256,256]`，输出 `logits: float32 [1,2]`，顺序 `lcd, wheel`；opset 12、固定 batch=1。
- 方向校正后整图 RGB 等比例缩放并居中补灰 128，PIL bilinear，256×256；像素 /255 后按 ImageNet mean/std 归一化。ONNX 元数据包含完整参数。
- 预训练权重下载地址及训练图片哈希、分组、设置保存在 `models/meter_type_cnn/training.json`；从 torchvision 的 `mobilenet_v3_small-047dcff4.pth` 初始化，下载时验证文件名哈希前缀。
- 本机 ONNX Runtime 1.31.0 CPU、4 线程，预热后单张推理中位数 **1.43 ms**；预处理加推理约 **40.58 ms**，再计入大图解码约 **124.51 ms**，均不含绘图/保存和会话创建。不能据此推断手机耗时。

`verify_meter_type.py` 从全部 86 张原图检查 NumPy 推理预处理与训练用 torchvision 预处理一致（最大差异为 0），并逐张比较对应留出折与最终模型的 PyTorch / ONNX 输出，最大 logits 绝对误差 2.44e-5，类别全部一致；验证详情在 `results/meter_type_training/runtime_verification.json`。训练导出时另通过 ONNX checker。

当前是闭集分类，总会选 lcd 或 wheel，分数未经正确率校准，没有“不是电表”拒识。没有实现滚轮读数，也没有测试手机或小程序执行兼容性。LCD 原有定位与识别模型继续独立使用。

## 查看结果

- `results/meter_type_training/held_out_errors.jpg`：11 张留出错例。
- `results/meter_type_training/held_out_overview.jpg`：全部 86 张留出预测。
- `results/meter_type_training/held_out_predictions.json` / `.csv`：每张图所在折、类别分数和预测。
- `results/meter_type_training/runtime_verification.json`：原图预处理和模型一致性验证。
- `results/meter_type/overview.jpg`、`classifications.json` / `.csv`：最终模型在全部原图上的回测。
- `results/meter_type/inputs/`：最终模型实际输入 PNG。

早期 MobileCLIP 零样本探索在最初 32 张图上仅 11/32，已保留为单独脚本与旧结果，不作为当前默认模型，也不与后来的 86 张分组验证直接比较。
