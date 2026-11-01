<h1 align="center">
  Vision Neuron Microscope<br>
  <sub>🔬 通过特征可视化探索视觉神经元</sub>
</h1>

<p align="center">
  <a href="source/inceptionv1/README.md"><img src="docs/assets/badges/model.svg" alt="Model: InceptionV1"></a>
  <a href="#overview"><img src="docs/assets/badges/neurons.svg" alt="Neurons: 5,804"></a>
  <a href="#reproduction-results"><img src="docs/assets/badges/modes.svg" alt="Modes: 5 visualizations"></a>
  <a href="https://vision-neuron-microscope.tart-olm-9383.chatgpt.site/?layer=mixed4c&amp;neuron=477&amp;compare=1"><img src="docs/assets/badges/viewer.svg" alt="Viewer: interactive HTML"></a>
</p>
<p align="center"><a href="README.md">English</a> · <b>简体中文</b></p>
<p align="center">
  <a href="#overview">📖 概览</a> ·
  <a href="#workflow">⚗️ 工作流程</a> ·
  <a href="#quick-start">🚀 快速开始</a> ·
  <a href="#reproduction-results">🧪 结果对比</a> ·
  <a href="#documentation">📚 文档</a>
</p>

## 📖 项目概览 <a id="overview"></a>

Vision Neuron Microscope 用于探索 InceptionV1 神经元学到的视觉特征。项目参考 [AlreadyClean](https://already-clean.goodfire.com/)，将优化图像、自然样例检索、激活热图和原文对比整合到一个交互页面中。

目录覆盖 **12 层、5,804 个 neuron**，支持 **Positive、Path-guided +、Diversity、Negative、Path-guided −** 五类结果。默认使用 **A 组设置、224 × 224 输入、1,024 步优化**。

**[打开交互式 Microscope ↗](https://ziheng-inception-microscope-a.tart-olm-9383.chatgpt.site/?layer=mixed4c&neuron=447&compare=1)**

## ⚗️ 工作流程 <a id="workflow"></a>

1. **选择 neuron。** 加载固定版本的 Lucent Inception5h，指定层和通道。
2. **准备自然样例。** 扫描 50,000 张 ImageNet 训练图片，获得 Top 16、Additional Examples、激活热图和路径锚点，并按分支感受野裁剪展示区域。
3. **优化图像。** 在 Fourier 空间中优化，结合拟合的颜色基、频段逐步解锁、模糊退火和 Gaussian 加权激活目标。
4. **查看对比。** 导出 HTML，浏览五类可视化、不同优化步数和 Nick / Goodfire 的参考结果。

| 类型 | 优化目标 |
|---|---|
| Positive | 最大化 post-ReLU 激活 |
| Path-guided + | 正向优化，并在早期加入自然样例路径引导 |
| Diversity | 同时优化四张正向图像，加入多样性目标 |
| Negative | 最小化有符号的 pre-ReLU 激活 |
| Path-guided − | 负向优化，并在早期加入自然样例路径引导 |

路径引导使用池化后的激活向量锚点。损失函数、颜色先验和目标定义见[方法说明](docs/methodology.md)。

## 🚀 快速开始 <a id="quick-start"></a>

### 1. 安装与准备模型

使用 Python 3.11+，在独立环境中安装与 GPU 兼容的 CUDA PyTorch。

```bash
git clone --branch hr-language-refactor https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope/vision
python -m pip install -e './vision-repro[dev]'
```

按照[模型准备](source/inceptionv1/README.md)将 Lucent 和权重放入 `source/inceptionv1/`，然后检查：

```bash
vision-microscope doctor --model-dir source/inceptionv1
vision-microscope catalog
```

复用其他位置的模型时，将 `VNI_MODEL_DIR` 设为该目录的绝对路径。

### 2. 生成并查看结果

```bash
vision-microscope optimize --layer mixed4c --neurons 447 \
  --variant positive --steps 1024 \
  --output vision-repro/outputs/mixed4c_447_positive
vision-microscope export \
  --runs vision-repro/outputs/mixed4c_447_positive \
  --output vision-repro/outputs/viewer
python -m http.server 8766 --directory vision-repro/outputs/viewer
```

打开 [localhost:8766](http://localhost:8766/?layer=mixed4c&neuron=447&compare=1)。每组配置使用独立的输出目录。

### 3. 生成完整五类结果

[运行指南](docs/reproduction.md)包含 ImageNet 扫描、锚点准备、全 neuron 队列、并发和断点续跑。Path-guided 和 Diversity 使用自然样例扫描得到的锚点。导出时通过 `--sample-run` 加入自然样例与热图，通过 `--reference-catalog` 加入官方对比。

导出支持 `--rf-snapshot`，为 Top 16、Additional Examples 和负向样例应用感受野裁剪。大量图片可使用独立的 [R2 图片存储](docs/hosting.md)。

## 🧪 结果对比 <a id="reproduction-results"></a>

四个 neuron，每个展示五类结果。**我们的设置：A · 224 × 224 · 1,024 步 · 默认 seed 0。** Diversity 展示四张图，点击图片可放大。

### mixed4c · 477

| | Positive | Path-guided + | Diversity | Negative | Path-guided − |
|---|---|---|---|---|---|
| Nick | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/477.png" width="144" alt="Nick 477 Positive"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/477.png" width="144" alt="Nick 477 Path-guided +"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/477.png" width="144" alt="Nick 477 Diversity"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/477.png" width="144" alt="Nick 477 Negative"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/477.png" width="144" alt="Nick 477 Path-guided −"></a> |
| Ours | <a href="docs/assets/mixed4c_477/positive.png"><img src="docs/assets/mixed4c_477/positive.png" width="144" alt="Ours 477 Positive"></a> | <a href="docs/assets/mixed4c_477/path_positive.png"><img src="docs/assets/mixed4c_477/path_positive.png" width="144" alt="Ours 477 Path-guided +"></a> | <a href="docs/assets/mixed4c_477/diversity.png"><img src="docs/assets/mixed4c_477/diversity.png" width="144" alt="Ours 477 Diversity"></a> | <a href="docs/assets/mixed4c_477/negative.png"><img src="docs/assets/mixed4c_477/negative.png" width="144" alt="Ours 477 Negative"></a> | <a href="docs/assets/mixed4c_477/path_negative.png"><img src="docs/assets/mixed4c_477/path_negative.png" width="144" alt="Ours 477 Path-guided −"></a> |

477 的 Diversity 左下图使用 seed 1，post-ReLU 激活为 8,031.94；其余三张保留 seed 0。

### mixed4c · 260

| | Positive | Path-guided + | Diversity | Negative | Path-guided − |
|---|---|---|---|---|---|
| Nick | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/260.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/260.png" width="144" alt="Nick 260 Positive"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/260.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/260.png" width="144" alt="Nick 260 Path-guided +"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/260.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/260.png" width="144" alt="Nick 260 Diversity"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/260.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/260.png" width="144" alt="Nick 260 Negative"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/260.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/260.png" width="144" alt="Nick 260 Path-guided −"></a> |
| Ours | <a href="docs/assets/mixed4c_260/positive.png"><img src="docs/assets/mixed4c_260/positive.png" width="144" alt="Ours 260 Positive"></a> | <a href="docs/assets/mixed4c_260/path_positive.png"><img src="docs/assets/mixed4c_260/path_positive.png" width="144" alt="Ours 260 Path-guided +"></a> | <a href="docs/assets/mixed4c_260/diversity.png"><img src="docs/assets/mixed4c_260/diversity.png" width="144" alt="Ours 260 Diversity"></a> | <a href="docs/assets/mixed4c_260/negative.png"><img src="docs/assets/mixed4c_260/negative.png" width="144" alt="Ours 260 Negative"></a> | <a href="docs/assets/mixed4c_260/path_negative.png"><img src="docs/assets/mixed4c_260/path_negative.png" width="144" alt="Ours 260 Path-guided −"></a> |

### mixed4c · 286

| | Positive | Path-guided + | Diversity | Negative | Path-guided − |
|---|---|---|---|---|---|
| Nick | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/286.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/286.png" width="144" alt="Nick 286 Positive"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/286.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/286.png" width="144" alt="Nick 286 Path-guided +"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/286.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/286.png" width="144" alt="Nick 286 Diversity"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/286.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/286.png" width="144" alt="Nick 286 Negative"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/286.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/286.png" width="144" alt="Nick 286 Path-guided −"></a> |
| Ours | <a href="docs/assets/mixed4c_286/positive.png"><img src="docs/assets/mixed4c_286/positive.png" width="144" alt="Ours 286 Positive"></a> | <a href="docs/assets/mixed4c_286/path_positive.png"><img src="docs/assets/mixed4c_286/path_positive.png" width="144" alt="Ours 286 Path-guided +"></a> | <a href="docs/assets/mixed4c_286/diversity.png"><img src="docs/assets/mixed4c_286/diversity.png" width="144" alt="Ours 286 Diversity"></a> | <a href="docs/assets/mixed4c_286/negative.png"><img src="docs/assets/mixed4c_286/negative.png" width="144" alt="Ours 286 Negative"></a> | <a href="docs/assets/mixed4c_286/path_negative.png"><img src="docs/assets/mixed4c_286/path_negative.png" width="144" alt="Ours 286 Path-guided −"></a> |

### mixed4c · 212

| | Positive | Path-guided + | Diversity | Negative | Path-guided − |
|---|---|---|---|---|---|
| Nick | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/212.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/212.png" width="144" alt="Nick 212 Positive"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/212.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/212.png" width="144" alt="Nick 212 Path-guided +"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/212.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/212.png" width="144" alt="Nick 212 Diversity"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/212.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/212.png" width="144" alt="Nick 212 Negative"></a> | <a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/212.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/212.png" width="144" alt="Nick 212 Path-guided −"></a> |
| Ours | <a href="docs/assets/mixed4c_212/positive.png"><img src="docs/assets/mixed4c_212/positive.png" width="144" alt="Ours 212 Positive"></a> | <a href="docs/assets/mixed4c_212/path_positive.png"><img src="docs/assets/mixed4c_212/path_positive.png" width="144" alt="Ours 212 Path-guided +"></a> | <a href="docs/assets/mixed4c_212/diversity.png"><img src="docs/assets/mixed4c_212/diversity.png" width="144" alt="Ours 212 Diversity"></a> | <a href="docs/assets/mixed4c_212/negative.png"><img src="docs/assets/mixed4c_212/negative.png" width="144" alt="Ours 212 Negative"></a> | <a href="docs/assets/mixed4c_212/path_negative.png"><img src="docs/assets/mixed4c_212/path_negative.png" width="144" alt="Ours 212 Path-guided −"></a> |

[图片参数与来源](docs/assets/provenance.json) · [原版 Microscope](https://already-clean.goodfire.com/microscope/inceptionv1.html?layer=mixed4c&neuron=477)

## 📚 文档导航 <a id="documentation"></a>

| 文档 | 内容 |
|---|---|
| [模型准备](source/inceptionv1/README.md) | 源码版本、权重与预处理 |
| [运行指南](docs/reproduction.md) | 优化、自然样例、批量生成与导出 |
| [方法说明](docs/methodology.md) | 图像参数化和五类目标 |
| [数据格式](docs/data-format.md) | 运行记录、检查点与网页资产 |
| [验证记录](docs/validation.md) | 单元测试与 GPU 检查 |

<details>
<summary><strong>仓库目录</strong></summary>

```text
vision-repro/
  src/vision_repro/     优化、检索、队列与命令入口
    defaults/          A 组配置与颜色统计
    web/microscope/    HTML viewer
  tests/               单元测试
  outputs/             生成结果（不入 Git）
source/inceptionv1/    模型来源与准备说明
docs/                 文档与示例图
```

</details>

运行软件测试：`pytest vision-repro/tests`。

## 🙏 致谢 <a id="acknowledgements"></a>

本项目参考 [Nick / Goodfire 的 AlreadyClean](https://already-clean.goodfire.com/)、[Lucent](https://github.com/greentfrapp/lucent) 和 [Lucid](https://github.com/tensorflow/lucid)。README 结构参考 [Neuron-Microscope](https://github.com/RUC-TrustworthyAI/Neuron-Microscope)。相关来源与许可见[第三方说明](THIRD_PARTY_NOTICES.md)。
