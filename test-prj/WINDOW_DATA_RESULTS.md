# 两类读数窗口数据合并

已生成 `datasets/meter_windows/data.yaml`，共 **601 张图片、609 个读数区框**。2026-10-19 已完成新窗口检测模型训练与 ONNX 导出；测试集及本地实测见 [WINDOW_TRAINING_RESULTS.md](WINDOW_TRAINING_RESULTS.md)。原 `display.onnx` 保留，配置和输出约定见 README。

类别固定为 `0: lcd`、`1: wheel`。按用户选择，`lcd` 包含 LCD、LED 及其他电子数字屏，保留血压计、温控器等通用数字仪表；`wheel` 仅为机械数字滚轮。目标框是一整条读数区域或原数据中标注的电子显示区域，不检测单个数字。

## 来源与标签转换

| 原始目录 | 输出图片 | lcd 框 | wheel 框 | 转换规则 |
| --- | ---: | ---: | ---: | --- |
| Digital Meter.v1i.yolov8 | 121 | 109 | 12 | 保留 display 坐标；人工复核后将其中 12 张滚轮图改为 wheel |
| DigitalGauge.v1i.yolov8 | 120 | 128 | 0 | 仅保留 display → lcd；丢弃 frame、gauge |
| Electricity meter.v5i.yolov8 | 99 | 77 | 22 | Digital meter → lcd，analog meter → wheel；这些标签本身框的是读数区 |
| meter-reader-detection.v5i.yolov8 | 261 | 188 | 73 | 保留 Reading 坐标，根据所属 Digital/Analog 整表框继承 lcd/wheel 类型 |
| 合计 | 601 | 502 | 107 | 609 个框 |

`meter-reader-detection` 的 Digital/Analog 只提供类别，不把其整表坐标作为检测目标；Serial No. 丢弃。原标签含分割多边形，输出取多边形的最小轴对齐包围框，统一为 YOLO 检测的五列格式。Reading 必须与类型父框有明确对应关系；缺失或类型冲突时拒绝猜测，人工复核例外写入独立清单。

四个原始 `data.yaml` 的来源、许可声明、版本与原类别记录在输出 `summary.json` 的 `source_metadata`。原下载目录没有修改。

## 人工复核与排除

助手检查了全部来源联系表，并放大查看了难辨样本。复核记录位于 `window_label_reviews.json`，每项绑定图片及标签文件的 SHA256，源文件发生变化时脚本会拒绝沿用记录。

- Digital Meter 的 12 个滚轮原图名：`102_jpg, 118_jpg, 5_jpg, 62_jpg, 68_jpg, 69_jpg, 75_jpg, 88_jpg, 89_jpg, 92_jpg, 97_jpg, 85_jpg`。
- detmet 有 5 个原图组缺少 Digital/Analog 类型标签，经目视确认是 LCD，补充类型，共 15 张增强图。组名为 `550_SML_3PH_2_jpg`、`565_Genus_3PH_jpg`、`582_Genus_1PH_3_jpg`、`585_SML_1PH_3_jpg`、`590_L-G_1PH_2_jpg`。
- `582_Genus_1PH_3_jpg` 的三个版本各有一个整表误标为 Reading 的大框，移除该框，保留真正的读数条。
- 排除 detmet 的 `815_L-G_jpg` 三张指针盘图片：它们不属于数字滚轮。
- 排除 DigitalGauge 两张仅有 frame、没有 display 的图片，以及 Electricity meter 两张只有数字、没有读数窗口父框的图片。它们属于窗口漏标，不当成负样本。

输入 608 张，保留 601 张，排除 7 张。具体文件名、原因见 `datasets/meter_windows/excluded.json`。人工检查不等同于独立人工标注验收。

## 划分与核验

| 集合 | 图片 | lcd 框 | wheel 框 |
| --- | ---: | ---: | ---: |
| train | 499 | 418 | 88 |
| valid | 60 | 48 | 13 |
| test | 42 | 36 | 6 |

保留原始数据划分，按“来源目录 + 去除 `.rf.` 后缀的原图名”聚合增强版本；同源图原本跨集合时，优先归入 test，其次 valid，避免将留出图片移回训练集。共有 6 张训练图片随同源组移动到留出集合。对所有保留图片还计算了解码后 RGB 像素哈希，未发现完全重复图片。

这能排除已知原图名及完全重复像素的跨集合泄漏，不能证明同一实体电表的不同照片完全独立。detmet 已含旋转等增强，601 张并不代表 601 个独立实物。

已执行并通过：

- 601 张导出图片与源图片哈希一致，源标签哈希匹配。
- 609 个输出框均为两类 YOLO 检测标签，坐标有限、面积为正、位于图像范围内，且与转换清单一致。
- 图片和标签文件清单完整，没有混入排除样本，没有已知同源组或完全相同像素跨集合。
- 四个回归用例：读数区继承类型而保留自身坐标、缺失/冲突父框拒绝赋类、移除复核确认的整表误标框、多边形转包围框及漏标图片排除。
- 目视检查合并后的两页 `preview_00.jpg`、`preview_01.jpg`，覆盖四个来源、两类及人工修正组。

输出 `manifest.json` 保存每张图片的原划分、最终划分、源文件哈希、输出路径、框坐标及对应源标签行；`verification.json` 保存核验结果。复现与核验命令见 README。后续训练使用 `datasets/meter_windows/data.yaml`。
