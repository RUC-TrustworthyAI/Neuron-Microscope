<p align="center">
  <img src="assets/neuron-microscope.png" alt="Neuron Microscope" width="100%">
</p>

<h1 align="center">Neuron Microscope</h1>

<p align="center">
  <strong>研究语言模型与视觉模型中的原生神经元。</strong>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/">语言模型报告</a> ·
  <a href="vision/README_zh.md">视觉模型可视化</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1">查看 Colorado 案例</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=zh">语言模型研究流程</a> ·
  <a href="#use-neuron-microscope">运行项目</a>
</p>

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/"><img src="assets/badges/pipeline.svg" alt="语言侧：神经元报告"></a>
  <a href="vision/README_zh.md"><img src="assets/badges/reports.svg" alt="视觉侧：神经元可视化"></a>
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=zh"><img src="assets/badges/live-demo.svg" alt="流程回放：语言侧"></a>
  <a href="#use-neuron-microscope"><img src="assets/badges/quickstart.svg" alt="使用：安装与运行"></a>
</p>

<a id="overview"></a>

Neuron Microscope 独立复现并研究 Nick Cammarata 的 [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) 中的神经元分析。项目包含 **Qwen3-32B 语言模型研究**与 **InceptionV1 特征可视化**。

| 模型 | 可以查看的内容 | 入口 |
| --- | --- | --- |
| **Qwen3-32B** | 激活证据、解释、受控测试、干预和尚未解决的案例。 | [交互式神经元报告](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/) · [研究代码](language/) |
| **InceptionV1／Lucent Inception5h** | 五种可视化目标、自然图像样例、热图和参考对比。 | [视觉模块指南与样例](vision/README_zh.md) · [优化代码](vision/vision-repro/) |

**数据：** 当前 50 份报告均使用 **Reasoning Mix 100M**。[数据指南](docs/data.md)介绍了语料来源、激活格式和读取工具。报告包含选定的上下文和测量结果；完整采样文本与激活归档尚无公开下载入口。

> [!IMPORTANT]
> **本项目独立开展复现。** 在 LLM 研究中，研究模型不会看到 AlreadyClean 的解释。原报告仅在最终发布时加入，用于对比。

## 目录

- [语言模型神经元报告](#language-model-neuron-reports)
- [视觉模型神经元可视化](#vision-model-neuron-visualizations)
- [我们检验了哪些 AlreadyClean 主张](#which-alreadyclean-claims-have-we-tested)
- [LLM 神经元研究流程](#the-investigation-pipeline)
- [如何评估报告中的证据](#evaluating-the-report-evidence)
- [当前案例中的观察](#observations-from-the-current-comparisons)
- [安装与使用](#installation-and-usage)
- [实测耗时与开销](#recorded-runtime-and-cost)
- [局限与未来工作](#limitations-and-future-work)
- [如何参与项目](#how-to-contribute)
- [团队相关研究](#related-research-from-our-group)
- [环境、数据与报告指南](#setup-data-and-report-guides)

## 语言模型神经元报告 <a id="language-model-neuron-reports"></a> <a id="side-by-side-interactive-comparisons"></a> <a id="explore-the-evidence"></a><a id="reports-and-walkthrough"></a><a id="live-demo"></a><a id="reproduction-results"></a>

### 双栏案例：Colorado 神经元

打开神经元链接后，左侧显示原报告，右侧显示我们的复现。滚动任意一侧会同步移动两份报告；关闭双栏后可以单独阅读复现报告。

[打开交互式对比 →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1">
    <img src="assets/colorado-comparison.jpg" alt="Colorado 原报告与 Reasoning Mix 100M 复现报告的双栏截图" width="92%">
  </a>
</p>

两份报告都发现了较强的 Colorado 正向响应，但对神经元具体计算什么提出了不同解释：

- **原报告：** gate 会在即将填入地区相关信息的位置打开，up 投影承载带符号的地区身份。
- **我们的复现：** 正极响应与 Colorado 地名、机构名有关，出现在地方政府、教育、医疗、联系信息和新闻上下文中。负极解释限于部分非 Colorado 机构的行政联系信息与引文格式。

在 Reasoning Mix 100M 研究中，`Contact block: {slot} End of contact.` 中的六组 Colorado 机构联系信息达到 **+188 至 +352**。每组取替换片段内绝对值最大的有符号响应，这六组的峰值都落在空格上。三组外州机构的响应为 **-26.5 至 +42.75**，使用 Colorado `720` 区号的通用诊所仍达到 **+165**。这些结果支持 Colorado 相关响应，但机构名、区号和格式各自的作用仍需分开检验。报告保留了完整文本、归因结果、干预和反例。

### 浏览当前全部报告

查看 **Reasoning Mix 100M 上的 50 个 Qwen3-32B 神经元报告**，导航页同时列出各报告的 Blind 和 Search 分数。

[打开报告目录 →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/">
    <img src="assets/report-directory.png" alt="Qwen3-32B 神经元对比目录" width="92%">
  </a>
</p>

每一行都可进入完整交互式报告，查看激活证据、样例、搜索、聚类、效应、测试、批评、独立评阅和开销记录。两列各有 49 个有效分数：**Blind 均值为 0.864，中位数为 0.891；Search 均值为 0.782，中位数为 0.808**。导航页也在列标题中展示这些统计值。

## 视觉模型神经元可视化 <a id="vision-model-neuron-visualizations"></a> <a id="vision-neuron-visualizations"></a>

[Vision 模块](vision/README_zh.md)已发布 InceptionV1 全部 **12 层、5,804 个 neuron** 的五类可视化：Positive、Path-guided +、Diversity、Negative 和 Path-guided −，共 **29,020 组结果、46,432 张优化图**。每个 neuron 还提供从 50,000 张 ImageNet 训练图片中检索的 Top 16、Additional Examples 和负向样例，支持感受野裁剪与激活热图。

[打开交互网页 →](https://vision-neuron-microscope.tart-olm-9383.chatgpt.site/?layer=mixed4c&neuron=477&compare=1)

<p align="center">
  <img src="vision/docs/assets/viewer-screenshot.png" alt="InceptionV1 Microscope：原始结果对比、五类可视化与最高激活样例" width="92%">
</p>

[五类结果对比 →](vision/README_zh.md#reproduction-results)

<table align="center">
  <thead>
    <tr><th scope="col">来源</th><th scope="col">Positive</th><th scope="col">Path-guided +</th><th scope="col">Diversity</th><th scope="col">Negative</th><th scope="col">Path-guided −</th></tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row"><a href="https://already-clean.goodfire.com/microscope/inceptionv1.html?layer=mixed4c&amp;neuron=477">Nick / Goodfire</a></th>
      <td><a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/positive/mixed4c/477.png" width="144" alt="Nick / Goodfire mixed4c:477, Positive"></a></td>
      <td><a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_positive/mixed4c/477.png" width="144" alt="Nick / Goodfire mixed4c:477, Path-guided +"></a></td>
      <td><a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/diversity/mixed4c/477.png" width="144" alt="Nick / Goodfire mixed4c:477, Diversity"></a></td>
      <td><a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/negative/mixed4c/477.png" width="144" alt="Nick / Goodfire mixed4c:477, Negative"></a></td>
      <td><a href="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/477.png"><img src="https://already-clean.goodfire.com/microscope/inceptionv1/fv/path_negative/mixed4c/477.png" width="144" alt="Nick / Goodfire mixed4c:477, Path-guided −"></a></td>
    </tr>
    <tr>
      <th scope="row">Ours</th>
      <td><a href="vision/docs/assets/mixed4c_477/positive.png"><img src="vision/docs/assets/mixed4c_477/positive.png" width="144" alt="Ours mixed4c:477, Positive"></a></td>
      <td><a href="vision/docs/assets/mixed4c_477/path_positive.png"><img src="vision/docs/assets/mixed4c_477/path_positive.png" width="144" alt="Ours mixed4c:477, Path-guided +"></a></td>
      <td><a href="vision/docs/assets/mixed4c_477/diversity.png"><img src="vision/docs/assets/mixed4c_477/diversity.png" width="144" alt="Ours mixed4c:477, Diversity"></a></td>
      <td><a href="vision/docs/assets/mixed4c_477/negative.png"><img src="vision/docs/assets/mixed4c_477/negative.png" width="144" alt="Ours mixed4c:477, Negative"></a></td>
      <td><a href="vision/docs/assets/mixed4c_477/path_negative.png"><img src="vision/docs/assets/mixed4c_477/path_negative.png" width="144" alt="Ours mixed4c:477, Path-guided −"></a></td>
    </tr>
  </tbody>
</table>

**mixed4c · 477**。我们的结果使用 A 配置、224 × 224 输入和 1,024 步优化。Positive 最大化 post-ReLU 激活；Negative 最小化有符号 pre-ReLU 激活。Diversity 每组包含四张图片。

模型使用固定版本的 **Lucent Inception5h**。[方法说明（英文）](vision/docs/methodology.md)介绍了优化目标、颜色先验与自然样例的采样方式。可按[安装与运行指南](vision/README_zh.md#quick-start)复现这些结果。

## 我们检验了哪些 AlreadyClean 主张 <a id="which-alreadyclean-claims-have-we-tested"></a>

原文讨论了多个层次的主张。下表记录语言模型研究中已经检查的内容与尚待回答的问题。

| 主张或问题 | 需要的实验证据 | 当前状态 |
| --- | --- | --- |
| 能否为 Qwen3 原生神经元给出连贯解释？ | 激活覆盖、独立构造的盲测文本、受控测试与干预。 | **初步证据：** Reasoning Mix 100M 的 50 份报告中，有些案例较清楚，有些部分一致，有些存在明显偏差。 |
| 原报告的解释能否独立复现？ | 独立盲测后再进行双栏比较。 | **结果因案例而异。** 50 份报告的目录列出了双方解释，以及 Blind 和 Search 结果。 |
| 语料选择是否改变可见行为？ | 在不同语料间匹配采样条件。 | 当前 50 份报告均使用 **Reasoning Mix 100M**，仅凭这组结果无法分离语料本身的影响。 |
| 弱激活对全局行为是否不重要？ | 跨任务、跨层的全网络阈值消融。 | **计划中**，尚无明确结论。 |
| 递归泛化能否解释看似混合的行为？ | 不同行为分支共享形成过程或因果机制的证据。 | **尚无结论。** 当前报告可以提供候选案例，尚未确立一般性结论。 |
| 原生神经元是否比 SAE 特征更有用？ | 匹配神经元与特征、数据、测试、干预预算和评估标准。 | **计划中。** 尚无一般性比较结论。 |

> [!IMPORTANT]
> **这张表将随新证据更新。新增证据应对应具体报告、模型、语料、干预或反例。**

## LLM 神经元研究流程 <a id="the-investigation-pipeline"></a> <a id="the-three-stage-investigation-pipeline"></a> <a id="how-an-investigation-works"></a><a id="workflow"></a>

### 跟随一份完整研究记录

[打开语言模型研究过程记录 →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=zh)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=zh#loops">
    <img src="assets/pipeline-walkthrough-zh.jpg" alt="语言模型 walkthrough 中三轮解释与盲标结果的截图" width="92%">
  </a>
</p>

这份记录展示了 **Reasoning Mix 100M 上的 L55_N11703** 从已有证据到最终报告的过程，包括各步提示词、输入节选、结构化输出、三版解释与盲标结果、实测实验和开销。选中的解释在构造文本和 GPU 实验前冻结。

### 当前可运行流程

`language/` 中的代码直接调用 LLM API，并组织目标模型测量。LLM 按约定的 JSON 格式返回解释、预测和实验输入；Python 负责选择证据、执行测量并生成 HTML。配置生成器提供下述设置。

1. **抽取证据。** 将 500 个位置均分到非空激活区间，并在各区间对应的完整语料中取样。只有某区间的全部位置仍不够配额时，才把余量分给其他区间。归档保留 gate、up 和 product 三个通道；解释模型只看 product 值，并结合上下文、极值 token、不同文档的代表样例、分位样本、token 选择性、峰值统计和直接 logit 效应。
2. **解释、检验并冻结。** 正负极各最多提出五条规则，规则说明与正文共同表达同一份解释。盲标模型在隐藏实测值的情况下预测采样位置，每批最多 100 点，五路并行。模型修订两次，每次读取紧前一版解释、500 点实测上下文和重点错误反馈，再完成一轮盲标。三版中，优先从方向准确率 **>90%** 的版本里选择覆盖率最高的；均未达标时，选择准确率最高的。解释及其对应标签一起冻结。某一极证据不足时，可以保留为未解释。
3. **执行测量。** Search 根据实测反馈进行三轮搜索，每轮请求六个区域，每区 16 条文本。模型设计四组受控测试，每组十个条件，测量后改进一次测试设计；头图来自这些受控测试。独立的 Blind 写作模型只看冻结解释与规则，为每个有解释支持的方向构造六条文本。Search 和 Blind 取全文最高或最低激活；受控测试测替换片段、共同后缀或指定 token。GPU 还批量执行归因、家族响应和干预。启用共享 worker 后，同一份常驻模型按队列服务多个运行，期间各运行的 API 请求可以并行。
4. **分析并发布。** LLM 在保留冻结解释与规则的前提下，分析支持证据、反例和尚未解释的现象，再由独立读者评估。固定代码生成交互式报告；提供参考快照时，在研究完成后加入双栏比较。

**Activation spectrum 与 Inside the neuron 共用一组样本，以及修订后冻结解释所对应的预测。** 打标模型读取完整上文，并将紧邻后文作为语言背景；解释无法明确预测方向时，归入 No visible cue。由于这些样本参与解释修订，预测一致程度用于提供开发反馈。独立生成的盲测文本与受控实验提供进一步检验。

具体见[流程说明](docs/workflow.md)、[测量细节](docs/method.md)与[运行配置](docs/running.md)。

## 如何评估报告中的证据 <a id="evaluating-the-report-evidence"></a> <a id="how-we-evaluate-an-explanation"></a>

当前目录展示 **Blind score** 和 **Search score**。两者均计算复现最高值／原报告最高值，以及 |复现最低值／原报告最低值|，再对可用比值求平均。分数可以超过 1；汇总均值和中位数时排除缺失值。Blind 使用仅根据冻结解释编写的文本，Search 则使用实测反馈。这些极值用于衡量相对原报告的响应强度；因果判断仍需结合以下实验。

| 维度 | 问题 | 报告中的证据 |
| --- | --- | --- |
| 覆盖度 | 能否解释正负两极和重要的剩余案例？ | 激活谱、样例、语义聚类、预测错误与 No visible cue 案例。 |
| 特异性 | 能否识别具体触发条件，并与周围主题区分？ | 受控替换、反例、搜索和盲测。 |
| 预测能力 | 能否预测形成解释时未使用的输入？ | 独立的最大／最小响应盲测仅根据解释生成候选文本，受控模板与家族测量提供补充证据。 |
| 因果相关性 | 改变神经元或输入，是否产生预期影响？ | 原生干预、消融、受控输入修改和输出测量。 |
| 稳定性 | 换样本、提示词或独立运行，解释是否仍成立？ | 在匹配条件下比较解释与测量结果。 |
| 不确定性 | 哪些现象仍未解释，或证据不足以区分？ | 明确的批评、替代解释、弱响应极和失败测试。 |

RelP 使用修改后的导数提供辅助归因证据，仅凭归因图无法确立原生因果效应。具体见[测量范围](docs/method.md#relp-and-intervention-scope)。

## 当前案例中的观察 <a id="observations-from-the-current-comparisons"></a> <a id="what-we-have-found-so-far"></a><a id="key-findings"></a>

### 概念相近，响应条件仍可能不同

以下案例来自当前 **Reasoning Mix 100M** 结果。结合解释与实验，可以同时看到共同概念和适用范围上的差异。

| 神经元 | 两份解释的共同点 | 仍有差异的部分 |
| --- | --- | --- |
| [L50_N3507](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L50_N3507/?compare=1) | 数学对象标签和变量引入可以产生正响应；有理数分式表达可以产生负响应。 | 我们将负极限于有理数、分式表达；原报告还推广到其他形式占位符，并强调几何对象的名称。 |
| [L24_N17907](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L24_N17907/?compare=1) | “Has/must be seen or experienced to be believed” 表达必须亲身体验。 | 我们的负极说明包括子词与异质残余效应；原报告将浅负极视为 token 偏移。 |
| [L57_N2065](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L57_N2065/?compare=1) | 算术相关段落可以产生正响应，部分研究机构署名产生负响应。 | 我们还纳入证明与定理上下文；原报告强调上下文中积累的算术信息，并区分多个负极模式。 |
| [L45_N16800](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L45_N16800/?compare=1) | 明确的双关表达可以产生正响应。 | 我们侧重双关旁注；原报告要求文本呈现词的双重读法，也覆盖名称来源清晰可见的情况。 |

L45_N16800 说明，仅有共同主题还不够。我们的语料最高值为 **95**，高于原报告语料最高值 **71.5**，但 Blind 最高值仅为 **9.125**，原报告为 **50.75**。这处差距提示我们，需要用对照实验区分双关标记与上下文中实际出现的双重读法。

### 用语料极值衡量构造文本的响应

Colorado 在 Reasoning Mix 100M 中的语料范围为 **-44.5 至 +308**。Search 达到 **-44.25 至 +296**，选中的头图达到 **-26.5 至 +352**，Blind 达到 **-47 至 +278**。头图测替换片段，Search 和 Blind 取全文极值；这些测量回答的问题不同，需要结合具体文本和评分规则阅读。

语料极值描述的是已有扫描，构造文本可以超过该范围。构造文本的高响应表明测试触及了较强激活；判断解释是否成立，还需查看受控对比和干预。

当前报告仍有三处不确定性：语料是否包含相关上下文，解释是否抓住了响应条件，以及写作模型能否依据解释构造强响应测试。需要匹配比较条件，才能区分这些影响。

## 安装与使用 <a id="use-neuron-microscope"></a><a id="installation-and-usage"></a><a id="quick-start"></a>

可以根据目标模型与证据选择入口。

| 方式 | 可以做什么 | 需要什么 |
| --- | --- | --- |
| **浏览结果** | 查看 LLM 报告、完整研究记录与视觉对比。 | 浏览器。 |
| **体验渲染器** | 生成并查看小型合成报告，用于检查软件与界面功能。 | Python；无需 API 密钥、Docker、GPU 或模型下载。 |
| **开展 LLM 研究** | 分析已有激活归档中的神经元，发布新报告。 | Linux、Python 3.10+、CUDA 主机、模型权重、激活数据和 LLM 端点。 |
| **可视化视觉神经元** | 优化图像并导出 InceptionV1 网页。 | Python 3.11+、独立环境、模型资产，以及用于优化的 CUDA 主机，见[视觉模块配置](vision/README_zh.md#quick-start)。 |

### 1. 体验离线渲染器

```bash
git clone --branch hr-language-refactor https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r language/requirements/cpu.txt

python language/examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

打开 <http://localhost:8000/>。页面使用合成内容，不能作为实验证据。

### 2. 配置完整研究

完整 pipeline 需要 Linux、Python 3.10+、支持 CUDA 的主机、本地 dense-Qwen3 权重、兼容的激活归档、仅含激活证据的报告包，以及 LLM 端点。安装测量依赖：

```bash
# 安装与 NVIDIA 驱动匹配的 PyTorch CUDA wheel。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r language/requirements/gpu.txt
```

使用配置生成器创建 `workflow.local.json`，填写神经元、本地模型、仅含激活证据的报告包、激活归档和显卡编号：

```bash
python language/examples/configure.py \
  --neuron L55_N11703 \
  --model /path/to/model \
  --report /path/to/activation-only/report \
  --activations /path/to/activation/archive \
  --gpus 0 1 2 3 \
  --output workflow.local.json
```

核对生成配置中的 `llm.model`、推理级别、端点地址和 token 单价。随附配置使用 `deepseek-flash` 和 `high` 推理；报告中的模型标识为 `deepseek-v4.1-flash-1017`。在环境中提供 `DEEPSEEK_API_KEY`，或通过 `NEURON_PAPER_API_KEY_FILE` 指向本地凭证文件。

```bash
NEURON_MICROSCOPE_GPU_WORKER=1 python -m language \
  --config workflow.local.json --output runs/example
python -m http.server 8000 --directory runs/example/report
```

添加 `--stop-after interpretation`，即可利用已有激活查看修订后的解释和两张标注图，此时尚未加载目标模型。使用相同配置再次运行时，会复用已完成的 API 响应，以及已保存的采样和测量记录。更改科学输入时，应使用新的输出目录。

并行研究多个神经元时，为每个运行分别准备配置和输出目录；共用 GPU 组的运行应使用相同的 `gpu_lock`。`NEURON_MICROSCOPE_GPU_WORKER=1` 启用共享常驻模型。随附配置的 `llm.max_concurrent_requests: null` 不设置应用层 API 并发上限。继续执行、双栏导出和执行记录见[运行说明](docs/running.md)。

## 实测耗时与开销 <a id="recorded-runtime-and-cost"></a> <a id="reproducibility-and-cost"></a>

每次 LLM 研究都会保留配置、API 请求与响应、token 费用估算、采样证据、解释与预测版本、测量和最终报告。新的解释仅根据目标激活证据形成，输入中不含旧解释或官方标签。

下面是语言模型 walkthrough 中 **Colorado／Reasoning Mix 100M** 研究的记录：

| 项目 | 记录值 |
| --- | --- |
| 被解释模型 | Qwen3-32B |
| 已有证据 | Reasoning Mix 100M；抽取 500 个位置，选中版本标注了 498 个位置 |
| 研究与辅助调用 | `deepseek-v4.1-flash-1017`，推理级别 `high` |
| GPU 组 | 4 × NVIDIA A100 40 GB |
| 实测执行时间 | 25 分 59 秒 |
| GPU 实验时间 | 实录 GPU 工作 5 分 15 秒；0.351 GPU 卡时 |
| API 流式响应时间 | 19 分 25 秒 |
| API 费用估算 | $0.6181，共 26 次完成的请求 |
| 记录中的总耗时 | 25 分 59 秒 |

GPU 时间包含记录到的模型加载和运算。API 流式时间对重叠区间只计算一次。GPU 与 API 工作可能重叠，因此两者时间不能直接相加得到执行时间。

以上来自单次实测。费用不含激活采集，并取决于所配置的模型端点和 token 单价。[已发布报告](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=0#interpretation-costs)与 [walkthrough](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=zh#report) 提供了具体记录。

## 局限与未来工作 <a id="limitations-and-future-work"></a>

### 当前局限

- 当前集合包含 50 个已有原报告的神经元，结果尚无法推广到 Qwen3-32B 的全部神经元。
- 完整激活归档尚无公开下载入口，外部用户目前需要提供兼容的输入数据。
- 离线演示使用合成内容，不能用于复现已发布的科学结果。
- 语料覆盖、提示词选择、证据展示与 LLM 随机性都可能改变解释。
- Blind 和 Search 分数比较激活极值，无法证明机制解释正确；对可用方向求平均时，也可能只包含单极。
- 负极和弱激活的解释往往不如强正向响应清楚。
- 我们尚未完成条件匹配的原生神经元与 SAE 比较。

### 未来工作 <a id="future-work"></a>

- [ ] 发布小型真实神经元证据包，让用户无需完整模型或激活归档即可重新生成一份科学报告。
- [ ] 从已有原报告的神经元扩展到模型中更广泛的随机样本。
- [ ] 系统衡量解释在独立运行、提示词、样本与语料间的稳定性。
- [ ] 按统一评估流程，比较条件匹配的原生神经元与 SAE 特征。
- [ ] 跨任务、跨层开展全网络阈值消融，检验弱激活对全局行为的影响。
- [ ] 检验递归泛化等 AlreadyClean 主要主张。
- [ ] 支持更多模型、层、语言和文体。

## 如何参与项目 <a id="how-to-contribute"></a> <a id="contribute"></a>

欢迎提供支持、限定或反驳某项解释的证据。

以下内容都有助于改进项目：

- **新的神经元案例：** 模型、层与神经元 ID、语料、采样过程和完整报告。
- **反例：** 测试上下文、评分位置、实测响应及所挑战的解释。
- **受控实验：** 能够区分两种合理解释的匹配输入。
- **新模型或语料：** 格式兼容的激活数据，以及来源与再分发条件。
- **评估改进：** 更可靠地检验覆盖、预测、因果相关性、稳定性或不确定性。
- **Pipeline 改进：** 更清晰的证据展示、更可靠的工具、更低的计算需求或更容易复现的流程。

可以通过 [GitHub Issue](https://github.com/RUC-TrustworthyAI/Neuron-Microscope/issues) 提交具体的神经元 ID、上下文、实验、结果或建议。可复现的失败案例与分歧同样有价值。

## 团队相关研究 <a id="related-research-from-our-group"></a> <a id="related-work-from-our-group"></a><a id="related-work"></a>

团队相关工作涵盖特征学习、概念分析、解释评估和 LLM 安全。发表信息也可参阅[王希廷老师的学院主页](https://ai.ruc.edu.cn/szdw/zrjs/wxt/index.htm)。

| 工作 | 研究关联 | 链接 |
| --- | --- | --- |
| **C²R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders** — Haoran Jin 等，ICML 2026 | 通过跨样本的一致性约束，缓解 SAE 中的特征分裂与吸收。 | [论文](https://arxiv.org/abs/2606.30609) · [代码](https://github.com/hr-jin/Cross-sample-Consistency-Regularization) |
| **Uncovering Safety Risks of Large Language Models through Concept Activation Vector** — Zhihao Xu 等，NeurIPS 2024 | 利用安全概念激活向量，研究语言模型的安全机制及其脆弱性。 | [论文](https://arxiv.org/abs/2404.12038) · [代码](https://github.com/SproutNan/AI-Safety_SCAV) |
| **Evaluating Readability and Faithfulness of Concept-based Explanations** — Meng Li 等，EMNLP 2024 | 形式化概念解释，提出可读性和忠实度的度量方法。 | [论文](https://aclanthology.org/2024.emnlp-main.36/) · [代码](https://github.com/hr-jin/Concept-Explanation-Evaluation) |
| **Do LLMs Know Their Vulnerable Scenarios?** — Ziheng Peng 等，arXiv 2026 | 利用 SAE 概念与因果归因，识别会削弱拒答的场景。 | [论文](https://arxiv.org/abs/2607.23496) |

## 环境、数据与报告指南 <a id="setup-data-and-report-guides"></a> <a id="documentation"></a>

| 指南 | 用途 |
| --- | --- |
| [LLM 安装与运行](docs/running.md) | 安装依赖，配置 GPU 与凭证，运行或继续一批实验。 |
| [LLM 输入数据](docs/data.md) | 准备激活归档和目标报告包。 |
| [LLM 方法与实现](docs/method.md) | 从流程找到代码，了解测量范围。 |
| [LLM 报告指南](docs/reports.md) | 查找某张图或某个章节的数据来源和生成阶段。 |
| [LLM 开发](docs/development.md) | 运行 CPU 软件检查与离线演示。 |
| [LLM 提示词与输出约定](language/workflow/) | 查看解释、打标、实验设计和报告分析的请求。 |
| [视觉模块与样例](vision/README_zh.md) | 安装 InceptionV1 模块，查看可视化结果并导出网页。 |

## 引用 <a id="citation"></a>

如果在学术工作中使用 Neuron Microscope，请引用本仓库，并注明具体使用的版本或报告：

```bibtex
@software{ruc_trustworthyai_neuron_microscope_2026,
  author = {Haoran Jin and Ziheng Peng and Xiting Wang},
  title = {Neuron Microscope: Reproducing and Stress-Testing Native Neuron Explanations},
  year = {2026},
  url = {https://github.com/RUC-TrustworthyAI/Neuron-Microscope}
}
```

## 致谢 <a id="acknowledgements"></a>

报告结构参考 [Goodfire 的神经元报告](https://already-clean.goodfire.com/)。RelP 适配自 [`TransluceAI/circuits`](https://github.com/TransluceAI/circuits)，来源版本记录在 [`language/tools/relp-source.json`](language/tools/relp-source.json)。

## 许可 <a id="license"></a>

LLM 实现采用 [MIT License](LICENSE)。引入的视觉模块保留其[来源与许可说明](vision/THIRD_PARTY_NOTICES.md)，上游尚未为其原创代码声明许可证。第三方组件保留各自原有许可证。
