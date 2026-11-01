# 数据格式与可追溯性

[English](data-format.md) · 简体中文

## 独立优化批次

```text
run/
  job.json           不可变任务身份：layer/channels/slot/config
  metadata.json      来源、权重摘要、版本、分支映射、目标定义、配置
  evaluation.json    initial/checkpoint/final 同定义、无随机扰动的分数
  history.json       优化期间的目标、损失、频段、模糊和学习率
  complete.json      成功完成标记、步数、耗时、峰值分配与保留内存
  resume.pt          FFT、Adam、随机状态；不发布到网页
  items/0/           第一项 neuron 的 init/checkpoint/final PNG
  items/1/           下一项 neuron，使用该批次的 seed 配置
```

Diversity 的每个 item 包含 4 张图拼接的预览以及各成员图像；evaluation 的 score 数组逐成员对应。网页预览分数为成员分数均值，完整数组在 JSON 中保留。

批量 campaign 在 `batches/<job>/` 下使用同一格式，`queue.sqlite` 保存 pending/running/done/unavailable/failed/split 等状态。`protocol.json` 保存默认配置、源码摘要与批次策略；`progress.json` 和 `telemetry.jsonl` 用于监测。

## 自然样例

scan 目录保留 `inputs.npy`、每层 scores/locations、`selected_images.jsonl` 和协议。`web/` 包含：

- `neurons/<layer>/<channel>.json`：Top 16 和 Additional Examples 的类别、真实样例标识、排名、分数、atlas 位置、热图 offset/shape/峰值。
- `atlases/*.webp`：8 张图片横排的显示资产。
- `heat/<layer>/<channel>.bin`：按 JSON offset 索引的 uint8 热图。
- `manifest.json`：资产相对路径与 SHA-256。
- `complete.json`：扫描数、neuron 数、样例数与 manifest 的精确字节摘要。

网页颜色深浅按每张图的响应范围归一化；负向显示 `max(-pre,0)`。不能把这些热图说成像素梯度归因。

## 静态导出

`data.json` 提供完整 5,804-neuron 目录、结果路径和样例来源。`our-results.json` 以 `layer:channel` 为键，`ours[variant][step]` 指向 PNG，`variantInfo` 保留实际配置、初末分数、种子和模型来源。只导入完整 batch，没有结果的 neuron 不生成虚假记录。

可选 `reference-catalog.json`：

```json
{
  "neurons": {
    "mixed4c:447": {"flags": "11111"}
  }
}
```

五个位置依次是 Positive、Path-guided +、Diversity、Negative、Path-guided −。`1` 表示已确认可用，`0` 表示未提供，`h` 表示已确认官网未公开。缺省所有标志为 `0`；不要将未经核验的可用性写为 `1`。参考图片 URL 指向官方域名，源码包和导出器均不自动复制图片。

每次 export 生成新目录和资产摘要，保持旧导出可恢复。生成目录不纳入 Git；`public_metadata` 会清除常见机器绝对路径，但发布自定义运行记录前仍应检查敏感内容。

## 感受野快照

`crop-examples` 输出独立、不可变的快照：

```text
rf_snapshot/
  examples/<release>/catalog.json
  examples/<release>/manifest.json
  examples/<release>/nodes/<layer>/<channel>.json
  images/<sha256>.webp
  upload-manifest.json
  complete.json
```

节点保存 `cropPolicy`、原始 `mapLocation`、`requestedRF`、实际 `cropBounds`、裁剪坐标内的 `activationPosition`，以及 `heatBase64` 中每样例 24 × 24 的 uint8 热图。`heatOffset` 按字节索引。负向热图仍使用 `max(-pre,0)`。

浅层裁剪图使用 8 × 6 atlas、每格 112 × 112；深层全图复用原有 8 × 1 atlas。前端读取 `atlasColumns`、`atlasRows`，而非假定只有一行。`atlasUrl` 在云端指向内容摘要，在静态导出时改写为本地相对路径。

压缩快照另外保留 `encoding-audit.json`：WebP quality 从 95 起，整张 atlas PSNR ≥36 dB、每格 ≥30 dB，否则保留无损图。压缩不会改变分数、排名、热图或裁剪坐标。

静态导出的 `data.json.rfExamples` 记录已裁剪节点；其余节点继续使用原始扫描结果。`publishedApi` 默认为 false，开启后才读取同源云端发布 API。优化图与自然样例使用独立发布指针。

## 完整发布与补跑合并

`prepare_completed_snapshot` 将原队列、已发布快照和 `accepted.json` 中的补跑结果合并。每个替代项保留自己的 seed、配置、源码摘要及初始／最终分数；`--complete-neurons-only` 仅更新五类结果齐全的 neuron。

`--include-zero-activations` 保留激活为零的原始图片；`--include-zero-diversity` 仅对 Diversity 开启这一行为。非有限分数、错误符号或不完整图片仍会被拒绝。`qualityFlags` 保留原始审计记录。

- `completeFiveModeNeurons`：五类图片均可显示的数量。
- `fullyValidFiveModeNeurons`：五类结果均满足严格激活条件的数量。
- `zeroActivationPolicy`：`reject`、`include_diversity` 或 `include_all`。

2026-11-01 的全量发布覆盖 5,804 个 neuron，其中 5,777 个满足严格激活条件，另外 27 个保留实际零激活图片与分数。发布记录见[部署说明](hosting.md#current-published-gallery)。
