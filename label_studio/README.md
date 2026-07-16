# Label Studio：扁平结构一期

导入 `flat_structure_v1.xml` 作为项目标注界面。每个任务只处理一个候选方块字。

一期可接受的结构只有：`⿰`（左右）、`⿱`（上下）、`⿲`（左中右）和 `⿳`（上中下）。如果图像不完整、包含多个字、是包围结构或含嵌套结构，选择相应质量标签或 `out_of_scope`，不要勉强填写 IDS。

自动预标注的 `character_hint`、`ids_hint` 和 `components_hint` 仅作提示，标注员必须以图像为准。建议所有 `out_of_scope`、`unreadable` 和人工修改 IDS 的任务进入第二人复核队列。
