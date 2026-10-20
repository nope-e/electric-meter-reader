# 滚轮识别模型候选（2026-10-20）

检索后的实测进展：VisionState 已下载、核验并在当前相同轮位上运行，见 [VISIONSTATE_RESULTS.md](VISIONSTATE_RESULTS.md)；Fast-OCR 尚未接入，原识别脚本默认模型未替换。当前失败同时涉及漏检、轮位切分和识别/状态解码，换识别器需要分层比较。

## 1. VisionState wheel reader：优先试的 ONNX 单轮位模型

- 作者模型卡及权重入口：https://huggingface.co/oleost/visionstate-wheel-reader
- 作者项目：https://github.com/oleost/VisionState
- 训练代码入口：https://github.com/oleost/VisionState/tree/beta/tools/wheelreader
- 模型卡列出的权重 `wheels-v1.onnx`，约 2.2 MB、56.5 万参数；Apache-2.0。
- 面向机械滚轮，每位输出 100 个状态的 logits（0.0～9.9）；仍需要定位每个轮位。
- 输入 N×1×48×32 float，0～1。模型卡预处理：逐像素 RGB 最小通道转灰度、双线性缩放至 32×48、1% 截断自动对比度、除以 255。与现有模型的 RGB NHWC 0～255 预处理不同，不能直接复用。
- 作者采用轮位进位关系联合解码。模型卡成绩的末位使用四舍五入，与本项目“未完成进位取前一数字”的口径不同，不能直接照抄数字解码。
- 模型卡自报 Word-Wheel 2,400 张测试图整串 97.7%（图片方向正确，使用作者解码及末位四舍五入）。这是作者结果，未在本地复现，不能作为相对于当前模型的胜出证据。

判断：现成 ONNX、体积小、覆盖半字，是本项目 ONNX Runtime 路线中最方便验证的新候选；但单换它无法纠正螺丝被切成首位、漏末位等定位错误。手机/微信运行库兼容和耗时均未验证。

## 2. Fast-OCR：输入整条读数区，绕开手工切位

- 作者论文与权重页：https://web.inf.ufpr.br/vri/publications/amr-unconstrained-scenarios/
- 论文：Towards Image-based Automatic Meter Reading in Unconstrained Scenarios: A Robust and Efficient Approach，IEEE Access，2021。
- 原架构：https://www.inf.ufpr.br/vri/databases/amr-unconstrained-scenarios/data/counter-recognition.cfg
- 原权重：https://www.inf.ufpr.br/vri/databases/amr-unconstrained-scenarios/data/counter-recognition.weights
- 类别：https://www.inf.ufpr.br/vri/databases/amr-unconstrained-scenarios/data/counter-recognition.names
- 作者说明：https://www.inf.ufpr.br/vri/databases/amr-unconstrained-scenarios/data/README.txt

这是电表抄表工作的识别阶段，YOLO/Darknet 模型在读数区内检测和识别各位数字，输入整条读数区，可绕开当前按白字轮廓切位的步骤。完整作者管线还包含四角预测、透视矫正与可读性分类；直接用现有 area 的矩形裁剪属于不同设置。

作者权重页明确公开架构、权重、数据描述和类别，原始格式是 Darknet，不是已经验证的 ONNX。本次浏览器读取 cfg/weights/names 时因二进制 content-type 无法预览，尚未下载核验内容、权重大小或转换兼容性。半字的输出语义及当前两类表的实际泛化需要检查源码/标签并实测。

判断：更适合验证“是否应取消手工切位”，但接入和 ONNX 转换成本高于 VisionState。不把论文里的完整管线成绩归于单独识别器或本项目。

## 建议的对照方式

1. 先给少量轮位准备相同且正确的裁图，让当前 class100、cont、VisionState 分别按各自官方预处理推理，比较完整字符、红字、过渡态和整数附近状态。当前错误切位保持原样用于另一组整链路对照。
2. 输出原始 logits/状态、当前约定的取整候选和预览，避免仅用模型分数或两模型一致率判断正确性。
3. 再测试 Fast-OCR 整条读数区输入，以判断取消手工切位能否改善当前失败。
4. 使用这 18 张调参后应另留新图验证，保留当前冻结基线。
