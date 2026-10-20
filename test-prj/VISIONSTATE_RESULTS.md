# VisionState 滚轮模型对照测试

后续新增的轮位检测与授权实拍适配见 [WHEEL_SLOT_RESULTS.md](WHEEL_SLOT_RESULTS.md)。新入口 `recognize_wheel_slots.py` 继续使用本报告核验过的 VisionState 权重、预处理和联合解码。以下保留固定旧切位的模型比较，不代表后续轮位模型的结果。

日期：2026-10-20。本轮测试 VisionState，Fast-OCR 尚未接入。结论：在同一批轮位裁图上，VisionState 与联合解码的候选明显优于原 class100，但窗口漏检、错误切位及整数附近取整仍导致失败。尚未修改原识别脚本的默认模型。

## 模型与测试设置

使用作者公开的 [VisionState wheel reader](https://huggingface.co/oleost/visionstate-wheel-reader)，文件 `models/visionstate/wheels-v1.onnx`，2,264,691 字节，opset 17 / IR 8。

- 固定模型版本：`42f32402973b3bd3de0258766404232b11e8216f`。
- SHA256：`37912f885b061a7b12298872fbdfa1b73ca0664363b293a55617b1d3680a98ac`，与固定版本仓库 LFS 元数据一致。
- Hugging Face 直连超时后经 hf-mirror.com 下载；镜像与实际哈希写入来源文件。
- [作者代码](https://github.com/oleost/VisionState) 固定 commit `9e5e4f54b309adbed159be244d3ec747f72efe61`。`vendor/visionstate/common.py` 为原始代码，Git blob 已核验；附 Apache-2.0 LICENSE 和 SOURCE.json。
- ONNX Runtime 1.31.0、CPUExecutionProvider、单线程，输入 `[N,1,48,32]` float32，输出 `[N,100]` logits。

复用 `results/test_wheel_onnx_v2/readings.json`：18 张图片的原 SHA256 均核验不变，13 张有轮位、共 65 个。逐位读取原始高分辨率 `slots/*.png`，没有拿旧模型的 20×32 输入放大冒充相同裁图。分类、area、轮位几何和失败状态不变。

预处理按作者实现：RGB 逐像素取最小通道 → PIL 双线性缩放 32×48 → autocontrast cutoff=1 → /255。解码使用作者联合进位约束、相邻状态 slack=1、共同偏移最多 ±2 档、偏移惩罚 0.3。保存独立 argmax、联合状态、全部 logits/概率、作者末位四舍五入候选以及本项目末位向下取整候选；主对照使用本项目“未完成进位取前一数字”的约定。无历史读数纠错，没有按测试图调参数。

本次比较的是“VisionState 加其配套联合解码”与“原 class100 加原解码”的完整识别阶段；并非仅替换 CNN 权重的消融实验。

## 结果

分类仍 18/18 wheel，有 area 14/18，13 张完成识别，4 张无 area、1 张无轮位。因此更换识别阶段不能改善这 5 张无输入的情况。

两页窗口预览与两页实际灰度输入均已目视检查。**助手初核**下，13 张有候选的图中，class100 有 2 张、cont 有 3 张、VisionState 有 10 张候选与目视参考/进位约定相符。这是可供用户复核的初步计数，不是独立标注集准确率；11 张转录了参考，另两张首位实际是螺丝且读数与可见数字明显不符，未强填半字真实串。完整记录为 `visual_review.json`。

| 文件名末四位 | 目视参考 | class100 | VisionState 主候选 |
| --- | --- | --- | --- |
| 9494 | 0179.2 | 0179.2 | 0179.2 |
| 9554 | 0179.2 | 0179.2 | 0179.2 |
| 9632 | 0179.2 | 0179.1 | 0179.1 |
| 0036 | 0707.1 | 0607.1 | 0707.1 |
| 0134 | 0707.1 | 0607.1 | 0707.1 |
| 0198 | 7892.1 | 6282.1 | 7892.1 |
| 0444 | 5369.8 | 5366.4 | 5369.8 |
| 0578 | 7617.2 | 6616.2 | 7617.2 |
| 0639 | 9498.4 | 9387.5 | 9498.4 |
| 0696 | 9498.4 | 9387.4 | 9498.4 |
| 0762 | 9498.4 | 6398.4 | 9498.4 |

仍失败的已识别图片：

- `1792476679923.jpg`、`1792476679989.jpg`：原切位把螺丝送入首位，漏掉真正小数轮。VisionState 仍输出错误串 `9220.8`、`0220.8`。
- `1792476679632.jpg`：红色 2 被预测为状态 1.9，按既定取整规则得到 1，仍输出 `0179.1`；作者四舍五入候选为 `0179.2`。不能因为这一张对上就全局改取整规则，`0198`、`0578` 的半字也会受影响。

复核更正：此前把 `1792476680578.jpg` 描述成“螺丝误入首位”有误。此次查看实际单轮位 PNG，可见首位是 7，原 class100 的错误来自识别/状态解码；已在旧 evaluation.json 保留原笔记并追加更正。本轮实际螺丝错位是 `9923`、`9989`。

## 验证与产物

19 项单元回归通过，包括红/黑轮位等价预处理、未完成进位、取整与作者四舍五入分离、跨位进位约束，以及使用 80 组随机输入对比作者训练端与部署端联合解码实现。ONNX 模型通过 checker，实际运行成功。手机/微信兼容性与耗时尚未验证。

产物位于 `results/test_wheel_visionstate/`：

- `readings.json` / `readings.csv`：18 张全部状态与三模型候选；JSON 保存 logits、概率、状态、偏移、裁图哈希、模型来源和耗时。
- `overview_00.jpg` / `overview_01.jpg`：保留原切位几何的窗口对照，`100` 为旧模型，`VS` 为新候选。
- `slots_00.jpg` / `slots_01.jpg`：新模型实际输入和独立 argmax 状态。
- `inputs/`：实际 32×48 灰度输入。
- `visual_review.json`：助手逐图初核与计数，供用户确认。

```powershell
# 已有固定权重时只做哈希核验，不联网
.\.venv\Scripts\python.exe prepare_visionstate_model.py
# 首次下载可选择 --host hf-mirror.com
.\.venv\Scripts\python.exe compare_visionstate.py --output results\visionstate_recheck
.\.venv\Scripts\python.exe -m unittest test_visionstate_wheel test_wheel_decoding test_area_selection test_merge_window_data -v
```
