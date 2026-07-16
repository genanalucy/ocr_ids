# 数据规范

项目使用 UTF-8 JSONL。每行是一个具体字形样本，而不是一个抽象码位。

## 必需字段

- `sample_id`：全局稳定标识，不得依赖行号重新生成；
- `ids`：当前规范 IDS，必须能解析为一棵完整树。

## 常用字段

- `image_path`：图像路径；
- `character`、`codepoint`：已编码字符及码位；
- `glyph_region`：`G/T/J/K/V/H` 等地区来源；
- `source`、`source_version`：数据来源及固定版本；
- `label_status`：`automatic`、`reviewed` 或 `gold`；
- `alternatives`：同一具体字形的其他可接受 IDS；
- `metadata`：字体、扫描来源、增强参数、数据划分等。

## 数据分层

```text
data/raw       原始下载数据，只读保存
data/interim   解析、冲突检测和待审核结果
data/processed 规范化记录、渲染图和固定拆分
```

所有派生记录必须能够追溯到 `source + source_version + sample_id`。不得覆盖原始 IDS；后续会增加 `ids_raw` 与规范规则版本字段。

`canonicalization-v1` 是完全自动、保守的第一版规则：目标字符必须是 CJK 汉字范围，IDS 的所有终端必须通过严格校验。规则不主动猜测部件等价性；未通过的记录写入 review JSONL，并保存 `review_reasons`。

## 图像约定

- 单个字形；
- 灰度或 RGB PNG；
- 默认 256×256，白底黑字；
- 保留原始纵横比例并居中；
- 数据增强参数进入元数据或由固定训练配置控制。

## 拆分约束

- `test_zero_char` 的整字不得出现在训练集；
- 其中所有叶部件必须在训练集出现；
- `test_closed` 可以与训练集共享整字，但必须使用未进入训练的字体样本；
- 真正未编码字符只进入最终测试集，不用于调参。
