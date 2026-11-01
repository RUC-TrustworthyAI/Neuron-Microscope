# 复现与批量生成

以下命令在仓库的 `vision/` 目录运行。使用 Python 3.11+ 和支持目标 GPU 的 PyTorch/CUDA；DGX Spark 需要兼容其架构的发行包，不能把通用 x86 wheel 当作安装方案。本仓库不会修改驱动或自动下载数据。先运行 `vision-microscope doctor`。

```bash
cd vision
```

## 1. 单 neuron 与 GPU 冒烟检查

```bash
vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant positive --steps 8 --output vision-repro/outputs/smoke
```

8 步仅验证预训练模型、真实梯度和输出记录，不用于评价语义复现。正式运行用 1024 步；`--seed` 默认 0。输出目录标识完整配置，重复同一命令会从检查点继续或读取完成结果；不同配置不能复用同一目录。

## 2. 自然样例扫描

准备自己已获授权的 `ILSVRC2012_img_train.tar`。保留原始嵌套 class tar 结构，不需要解压。首次使用应校验完整文件，原始训练包的 MD5 为 `1d675b47d978889d74fa0da5fadfb00e`：

```bash
md5sum /absolute/path/to/ILSVRC2012_img_train.tar
```

核对成功后，将日期、校验算法和实际摘要记录到同目录的 `ILSVRC2012_img_train.tar.verified` 文件。不要为了绕过检查创建空标记。扫描器要求这个既有验证记录；不会在每次扫描时重新读取约 138 GiB 的归档来校验。该 MD5 用于识别原始归档，不是来源可信性证明。

```bash
vision-microscope scan \
  --archive /absolute/path/to/ILSVRC2012_img_train.tar \
  --output vision-repro/outputs/imagenet_scan --batch-size 24 --workers 4
```

扫描范围固定为每类 50 张、共 50,000 张训练图片。保存裁剪输入、全部通道分数、采样清单、热图和网页样例。已有扫描可复用，禁止覆盖；预留足够磁盘空间。`scan` 读取数据，不训练模型。

## 3. 路径锚点和五类结果

```bash
vision-microscope anchors --scan vision-repro/outputs/imagenet_scan \
  --output vision-repro/outputs/all_neurons/anchors

vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant path_positive --steps 1024 \
  --anchors vision-repro/outputs/all_neurons/anchors/mixed4c.pt \
  --output vision-repro/outputs/mixed4c_447_path_positive
```

将 `--variant` 换为 `positive`、`diversity`、`negative` 或 `path_negative`，使用各自唯一的输出目录。Positive 和 Negative 不需要 anchors；路径和 Diversity 需要真实自然样例锚点（后者用其分数标定损失权重）。仅加载自己信任的检查点/锚点文件；优化恢复文件包含 Python/RNG 状态，不应接受不可信上传。

## 4. 全量生成

```bash
vision-microscope campaign init --root vision-repro/outputs/all_neurons --batch-images 8
vision-microscope campaign run --root vision-repro/outputs/all_neurons --workers 1
vision-microscope campaign status --root vision-repro/outputs/all_neurons
```

共 29,020 个 neuron-variant 请求，Diversity 每项含 4 张图。`--batch-images` 必须是正的 4 的倍数；Diversity 每批 neuron 数为它的四分之一。先以 1 个 worker 测量，再按实际剩余资源调高。独立 neuron 的损失求和，避免改变 batch 大小时缩放每个目标的梯度。

监督进程记录 GPU 和系统内存、减少内存不足时的并发、只管理自己启动的子进程。根据本机实测显存和吞吐量调整 worker 数量。不要同时开启两个 supervisor；目录锁会拒绝第二个。不要在运行中修改被协议记录的代码文件。重启同一 campaign 可恢复未完成任务，缺少有效锚点的任务标为 unavailable。

## 5. 感受野裁剪与网页导出

使用已有扫描生成裁剪图和对齐热图；此步骤只使用 CPU：

```bash
vision-microscope crop-examples \
  --scan vision-repro/outputs/imagenet_scan \
  --model-source source/inceptionv1/lucent/lucent/modelzoo/inceptionv1/InceptionV1.py \
  --output vision-repro/outputs/rf_lossless --workers 2
```

可加 `--targets conv2d0:0 mixed3a:0 mixed4c:447 mixed5b:0` 先检查浅、中、深层。
每个 neuron 保留 Top 16、两组 Additional Examples 和负向 Top 16，共 48 个样例。
裁剪边长由所在分支决定，上限 224；靠边的裁剪框平移到输入内。深层感受野覆盖整张输入时保留全图。

可选生成较小的网页预览，原始无损快照保持不变：

```bash
python -m vision_repro.compress_rf_site_snapshot \
  vision-repro/outputs/rf_lossless vision-repro/outputs/rf_preview
```

压缩逐图检查质量；不达标时保留无损版本。将裁剪快照合并到静态网页：

```bash
vision-microscope export --runs vision-repro/outputs/all_neurons \
  --sample-run vision-repro/outputs/imagenet_scan \
  --rf-snapshot vision-repro/outputs/rf_preview \
  --output vision-repro/outputs/site
python -m http.server 8766 --directory vision-repro/outputs/site
```

导出仅包含完成的批次及通过摘要验证的样例资产，不复制 checkpoint、原始 ImageNet 归档。未计算结果显示占位。导出目录必须是新的；传入多个 runs 时，同一 neuron/variant 重复会报错，请显式选择一组结果。

可选 `--reference-catalog /path/to/reference-catalog.json`：提供已验证的官方可用图片索引以开启参考对比；不下载或复制官方 PNG。无此索引时，参考图保留占位。格式见 [数据格式](data-format.md)。本地浏览器仍可通过链接访问官方 neuron 页面。

完整静态目录可交给静态文件服务器。全部图片可能占用数 GiB，不适合直接塞进 GitHub 源码仓库。发布前检查 ImageNet/第三方资产的使用和再分发权限；上传源码不等同于授权公开数据集照片。

云端分批发布使用 `--published-api` 和独立图片存储，见 [部署说明](hosting.md)。普通静态页面无需这个参数。

## 6. 颜色统计与扩展对照

包内包含已拟合的数值统计，不含拟合图片。若需要重新拟合自己的归档：

```bash
python vision-repro/scripts/fit_imagenet_color.py \
  --archive /absolute/path/to/ILSVRC2012_img_train.tar
```

新结果写入独立 `outputs/color_statistics/` 时间戳目录；通过 optimize 的 `--fitted-color` 指定其中的 `statistics.json`。默认 A 组参数位于包内 `defaults/A.json`；改变配置后务必使用新的运行目录并记录修改，不能将结果标为原始 A 组。
