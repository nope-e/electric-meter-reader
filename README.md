# 电表读数识别

当前流程：整图分类 `digital/wheel` → 单类 `area` 读数区检测 → LCD 数字识别，或滚轮完整轮位检测 + VisionState 状态识别。推理使用 ONNX Runtime。

- [models/README.md](models/README.md)：5 个部署 ONNX、输入输出、前后处理、模型来源与独立推理入口。
- [test-prj/README.md](test-prj/README.md)：训练、数据合并、历史实验及运行说明。
- [test-prj/ALL_LOCAL_RESULTS.md](test-prj/ALL_LOCAL_RESULTS.md)：全部本地自拍照片的统一回测结果。

## 数据集

`datas/` 中当前的自拍原图及下载数据集均纳入版本管理；原始图片不压缩、不重新编码。本地参考读数和人工框复核记录在 `test-prj/*reference.json`、`test-prj/*reviews.json`。生成的训练缓存、派生数据目录、实验输出和本机配置不纳入版本管理。

### 自拍数据

以下图片由项目使用者自行拍摄，共 104 张：

| 目录 | 图片数 | 内容 |
| --- | ---: | --- |
| `datas/lcd/` | 33 | 数字屏电表 |
| `datas/wheel/` | 53 | 滚轮电表，包含 5 位和 6 位布局 |
| `datas/test_wheel/` | 18 | 补充滚轮电表照片 |

其中部分照片已经参与类型分类或轮位检测训练，`test_wheel` 也用于适配；整批回测不能当作独立泛化测试。完整读数评分与参考不确定样本的处理见上述回测报告。

### 下载数据来源与署名

以下信息按下载包自带的 `data.yaml`、`README.dataset.txt` 和 `README.roboflow.txt` 记录。保留原包的来源说明、图片、YOLO 标注及 train/valid/test 划分；表中数量为当前目录内图片数。四个下载包共 608 张图片。

| 本地目录 | 发布者 / workspace | 数据集版本及来源 | 图片数 | 原包声明许可 |
| --- | --- | --- | ---: | --- |
| `datas/Digital Meter.v1i.yolov8/` | `piyush-kumar-uou3u` | [Digital Meter v1](https://universe.roboflow.com/piyush-kumar-uou3u/digital-meter-gzxcg/dataset/1) | 121 | CC BY 4.0 |
| `datas/DigitalGauge.v1i.yolov8/` | `abc-gauge-reader` | [DigitalGauge v1](https://universe.roboflow.com/abc-gauge-reader/digitalgauge/dataset/1) | 122 | CC BY 4.0 |
| `datas/Electricity meter.v5i.yolov8/` | `sipl` | [Electricity meter v5](https://universe.roboflow.com/sipl/electricity-meter-lpx08/dataset/5) | 101 | CC BY 4.0 |
| `datas/meter-reader-detection.v5i.yolov8/` | `detmet` | [meter-reader-detection v5](https://universe.roboflow.com/detmet/meter-reader-detection-4enbh/dataset/5) | 264 | CC BY 4.0 |

许可原文：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。这些第三方数据保留原作者署名及许可声明；本项目没有将其改为自拍数据或重新授予统一许可。Roboflow 原包中已有的缩放和数据增强信息见各目录的 `README.roboflow.txt`。

训练时对读数区标签进行筛选、合并为单类 `area`，通用数字屏也纳入；人工修正和合并逻辑分别保存在 `test-prj/window_label_reviews.json`、`test-prj/merge_window_data.py`、`test-prj/prepare_area_data.py`。这些处理生成派生训练目录，本次提交不改写下载包内的原标注。轮位检测数据由项目准备脚本和人工复核记录生成，具体版本见 [WHEEL_SLOT_RESULTS.md](test-prj/WHEEL_SLOT_RESULTS.md)。

## 快速推理

在已有项目环境中，于仓库根目录执行：

```powershell
.\test-prj\.venv\Scripts\python.exe models\reference\pipeline.py --verify-only
.\test-prj\.venv\Scripts\python.exe models\reference\pipeline.py --input datas\lcd\1792403667245.jpg --output models\runs\single.json
```

新环境安装 `models/requirements.txt` 后，可用 `python models/reference/pipeline.py` 调用同一入口。输出文件应使用新路径。当前已验证桌面 CPU；微信小程序端尚未实测。
