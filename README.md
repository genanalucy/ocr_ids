# OCR-IDS：方块字图像到 IDS 结构识别

OCR-IDS 研究单个方块字图像与 IDS（Ideographic Description Sequence，表意文字描述序列）之间的结构化转换。目标不是普通整字分类，而是把字形解析为部件和空间布局，并探索对未见整字的组合泛化能力。

> 当前为一期研究原型，不是通用 OCR 产品。模型在合成字体域表现良好，但真实截图、扫描和手写输入仍需真实标注数据微调。

## 一期范围

一期只处理四种非嵌套的扁平结构：

| 结构 | IDS | 例子 |
|---|---|---|
| 左右 | `⿰` | 识 → `⿰讠只` |
| 上下 | `⿱` | 苗 → `⿱艹田` |
| 左中右 | `⿲` | 示例结构 |
| 上中下 | `⿳` | 示例结构 |

包围结构、嵌套结构、多字图、截断图和低质量图均超出当前范围，应该拒识或送入人工复核队列。

## 系统能力

```text
单字图像
  → 背景/颜色无关的前景提取、自动裁切、居中补白
  → DINOv2 ViT-S 图像编码器
  → Transformer IDS 解码器
  → 四结构受限解码
  → IDS、结构树、置信度、输入质量告警
```

- 支持 PNG、JPEG、WebP 单字图上传；
- 自动处理彩色字、深浅背景、尺寸和留白差异；
- 返回归一化预览，以及贴边、疑似多字、低对比度等提示；
- 训练、数据处理和前端可在远端 GPU 服务器运行；本机只保存代码和小样例。

## 当前训练成果

主模型为 `flat-structure-v1`：DINOv2 ViT-S/14 + 6 层 Transformer Decoder（384 hidden dim、6 heads），仅允许 `⿰`、`⿱`、`⿲`、`⿳` 作为根结构。

| 项目 | 结果 |
|---|---|
| 训练数据 | 373,648 张合成训练图；23,132 张验证图 |
| 训练硬件 | 4 × NVIDIA L20（GPU 4–7） |
| 训练配置 | bf16、AdamW、每卡 batch 96、40 epoch、全局 batch 384 |
| 同字体域留出字体抽测 | 20/20 IDS 完全匹配 |
| 合成未见整字抽测 | 18/20 IDS 完全匹配 |

以上是小样本合成图像抽测，不是对真实世界输入的正式精度。完整训练过程、局限和下一阶段计划见 [训练报告](docs/TRAINING_REPORT_2026-07-16.md)。

## 快速开始

### 本地开发

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

ocr-ids parse '⿰讠只'
ocr-ids validate '⿱艹田'
pytest
```

### 运行前端与推理 API

训练完成后，在服务器启动。checkpoint 同目录必须有 `vocab.json`。

```bash
cd /home/hzh/ocr_ids
source /home/hzh/ocr_ids_runtime/env.sh
"$HOME/.local/bin/uv" pip install --python .venv/bin/python -e '.[web]'

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m ocr_ids.web \
  --checkpoint "$OCR_IDS_RUNS_ROOT/dinov2_vits_flat_v1/last.pt" \
  --host 0.0.0.0 --port 8010
```

浏览器访问 `http://服务器地址:8010/`。接口：

```text
GET  /api/status
POST /api/predict    # multipart/form-data，字段名 image
```

## 数据流水线

标签来自 CJKVI IDS 数据，经严格解析、规范化和字体覆盖筛选后，用 Noto CJK 的 18 个字体 face 渲染为 224×224 单字图。

```text
CJKVI IDS
  → 规范化 / 严格校验
  → 字体覆盖筛选
  → 多字体渲染
  → 按整字隔离的 train / validation / zero-char 划分
  → flat-structure-v1 过滤
  → DDP 训练
```

构建最小渲染数据：

```bash
pip install -e '.[data]'
python scripts/render_dataset.py \
  --labels examples/labels.jsonl \
  --font /path/to/NotoSansCJK-Regular.ttc#2 \
  --output data/processed/rendered
```

TTC/OTC 字体集合用 `路径#face序号` 指定具体 face。

## 训练

安装训练依赖：

```bash
pip install -e '.[train]'
```

在 GPU 4–7 训练当前一期模型：

```bash
OCR_IDS_GPU_IDS=4,5,6,7 OCR_IDS_NPROC=4 \
OCR_IDS_TRAIN_CONFIG=configs/train_flat_l20x4.yaml \
bash scripts/remote_train_l20x6.sh
```

查看训练日志：

```bash
tail -f /home/hzh/ocr_ids_runtime/runs/flat-v1/train.log
```

## Label Studio 与真实数据闭环

一期标注配置在 [label_studio/flat_structure_v1.xml](label_studio/flat_structure_v1.xml)。它让标注员审核：

- 图像是否为完整、可判定的单字；
- 四种扁平结构之一，或“超出范围”；
- Unicode 字符、规范 IDS、视觉部件与备注。

真实截图、扫描和手写数据应先自动预标注，再由人工复核。不要把没有明确使用许可的网络图片直接混入训练集。

## 项目结构

```text
src/ocr_ids/       IDS、模型、预处理和 Web API
scripts/           数据导入、渲染、划分与训练脚本
configs/           可复现实验配置
label_studio/      一期人工标注配置
docs/              方案、数据规范、训练报告
examples/          最小标签样例
tests/             单元测试
```

## 数据与模型存储策略

本仓库只跟踪代码、配置、文档、测试与最小样例。正式数据集、字体、checkpoint、缓存和运行日志不进入 Git，默认位于训练服务器：

```text
/home/hzh/ocr_ids_runtime/datasets
/home/hzh/ocr_ids_runtime/models
/home/hzh/ocr_ids_runtime/cache
/home/hzh/ocr_ids_runtime/runs
```

同步代码到服务器不会上传本地数据目录：

```bash
bash scripts/remote_sync.sh
```

## 已知限制

1. 训练图仍主要是 Noto 合成字体；真实截图泛化尚不足；
2. 当前只支持四类扁平结构；
3. 置信度尚未在真实数据上校准；
4. 尚需建立完整的真实标注测试集，报告 exact match、部件准确率、拒识率和校准指标。

后续工作见 [一期实施方案](docs/PHASE1_PLAN.md) 和 [训练报告](docs/TRAINING_REPORT_2026-07-16.md)。
