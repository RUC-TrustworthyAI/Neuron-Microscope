<p align="center">
  <img src="assets/neuron-microscope.png" alt="Neuron Microscope" width="100%">
</p>

<h1 align="center">Neuron Microscope</h1>

<p align="center">
  <strong>原生 LLM 神经元能被解释吗？这些解释能经受受控实验的检验吗？</strong>
</p>

<p align="center">
  一个以证据为基础的开放框架，用于复现和检验神经元解释。
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://neuron-microscope-papers.tangysiren10.chatgpt.site/">浏览神经元报告</a> ·
  <a href="https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/">查看 Colorado 案例</a> ·
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh">跟随一次研究</a> ·
  <a href="#use-neuron-microscope">运行项目</a>
</p>

<p align="center">
  <a href="docs/method.md"><img src="assets/badges/model.svg" alt="模型：dense Qwen3"></a>
  <a href="skills/README.md"><img src="assets/badges/skills.svg" alt="Agent skills：三个阶段"></a>
  <a href="docs/reports.md"><img src="assets/badges/reports.svg" alt="报告：交互式 HTML"></a>
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh"><img src="assets/badges/live-demo.svg" alt="展示：在线"></a>
  <a href="#use-neuron-microscope"><img src="assets/badges/quickstart.svg" alt="使用：安装与运行"></a>
</p>

<a id="overview"></a>

Neuron Microscope 独立复现并研究 Nick Cammarata 的 [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) 中的神经元分析。我们从 **Qwen3-32B** 的原生 MLP 神经元入手，在报告中展示激活证据、假设、受控实验、因果干预、修订过程与尚未解决的失败案例。

**当前提供：** [研究代码与 agent skills](skills/README.md)、[五套语料的来源、采样流程和读取工具](docs/data.md#corpora-used-in-the-comparisons)，以及 [10 份交互式神经元对比](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)。报告包含选定的上下文和测量结果；完整采样文本与激活归档尚无公开下载入口。

> [!IMPORTANT]
> **本项目独立开展复现。** 研究 agent 在调查过程中不会看到 AlreadyClean 的解释。原报告仅在最终发布时加入，用于对比。

## 目录

- [交互式双栏报告对比](#side-by-side-interactive-comparisons)
- [我们检验了哪些 AlreadyClean 主张](#which-alreadyclean-claims-have-we-tested)
- [三阶段神经元研究流程](#the-three-stage-investigation-pipeline)
- [如何评估报告中的证据](#evaluating-the-report-evidence)
- [当前案例中的观察](#observations-from-the-current-comparisons)
- [安装与使用](#installation-and-usage)
- [实测耗时与开销](#recorded-runtime-and-cost)
- [局限与未来工作](#limitations-and-future-work)
- [如何参与项目](#how-to-contribute)
- [团队相关研究](#related-research-from-our-group)
- [环境、数据与报告指南](#setup-data-and-report-guides)

## 交互式双栏报告对比 <a id="side-by-side-interactive-comparisons"></a> <a id="explore-the-evidence"></a><a id="reports-and-walkthrough"></a><a id="live-demo"></a><a id="reproduction-results"></a>

### 双栏案例：Colorado 神经元

打开神经元链接后，左侧显示原报告，右侧显示我们的复现。滚动任意一侧会同步移动两份报告；关闭双栏后可以单独阅读复现报告。

[打开交互式对比 →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/)

<p align="center">
  <a href="https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/">
    <img src="assets/colorado-comparison.jpg" alt="Colorado 神经元的原报告与复现报告" width="92%">
  </a>
</p>

两份报告都发现了较强的 Colorado 正向响应，但对神经元具体计算什么提出了不同解释：

- **原报告：** gate 会在即将填入地区相关信息的位置打开，up 投影承载带符号的地区身份。
- **我们的复现：** 神经元广泛响应 Colorado 的地名、机构和地方新闻上下文。

在 SlimPajama 100M 实验中，使用句式 `Officials in {city} approved the annual budget this week.`，在单个 token ` approved` 上评分，四个 Colorado 城市产生 **+22.0 至 +43.3** 的激活，其余六个城市均不超过 **+4.75**。对比页提供了上下文、受控替换、归因结果、干预，以及仍未充分解释的负极。

### 浏览当前全部报告

[打开报告目录 →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)

<p align="center">
  <a href="https://neuron-microscope-papers.tangysiren10.chatgpt.site/">
    <img src="assets/report-directory.jpg" alt="Qwen3-32B 神经元对比目录" width="92%">
  </a>
</p>

每一行都可进入完整交互式报告，查看激活证据、样例、搜索、聚类、效应、测试、解释历史和开销记录。

## 我们检验了哪些 AlreadyClean 主张 <a id="which-alreadyclean-claims-have-we-tested"></a>

原文讨论了多个层次的主张。下表区分本项目已经检查的内容与尚待回答的问题。

| 主张或问题 | 需要的实验证据 | 当前状态 |
| --- | --- | --- |
| 能否为 Qwen3 原生神经元给出连贯解释？ | 激活覆盖、留出样例、受控测试与干预。 | **初步证据：** 有些案例较清楚，有些只有部分一致或存在明显偏差。 |
| 原报告的解释能否独立复现？ | 独立盲测后再进行双栏比较。 | 首批 10 个案例中，**结果因案例而异**。 |
| 语料选择是否改变可见行为？ | 在不同语料间匹配采样条件。 | L51_N20229 的**观测范围存在差异**；分离语料本身的影响仍需匹配采样与上下文处理。 |
| 弱激活对全局行为是否不重要？ | 跨任务、跨层的全网络阈值消融。 | **本项目尚未检验。** |
| 递归泛化能否解释看似混合的行为？ | 不同行为分支共享形成过程或因果机制的证据。 | **尚无结论。** 当前报告可以提供候选案例，尚未确立一般性结论。 |
| 原生神经元是否比 SAE 特征更有用？ | 匹配神经元与特征、数据、测试、干预预算和评估标准。 | **计划中。** 尚无一般性比较结论。 |

这张表将随新证据更新。新增证据应对应具体报告、模型、语料、干预或反例。

## 三阶段神经元研究流程 <a id="the-three-stage-investigation-pipeline"></a> <a id="how-an-investigation-works"></a><a id="workflow"></a>

### 跟随一份完整研究记录

[打开研究过程记录 →](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh)

<p align="center">
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh">
    <img src="assets/pipeline-walkthrough-zh.jpg" alt="Neuron Microscope 研究过程记录" width="92%">
  </a>
</p>

这份记录展示 Colorado 神经元从已有激活证据到解释、受控实验、修订和发布的全过程，包括提示词、工具调用、测量、中间报告和最终批评。该记录使用 Diverse 100M，上面的对比案例使用 SlimPajama 100M。

### 从证据到报告

<p align="center">
  <a href="assets/workflow.png">
    <img src="assets/workflow.png" alt="解释、实验与修订流程" width="100%">
  </a>
</p>

*图中的数字用于说明步骤。实测输入、评分位置和数值见链接中的报告与研究记录。*

1. **准备证据。** 从已有激活归档中按激活区间抽取 token 位置，保留上下文，以及实测 gate、up 和 product 值。
2. **解释与打标。** 新的 agent 容器读取目标神经元的证据，提出初步解释并定义响应分组。固定工具逐样本打标，打标时隐藏待评样本的实测响应。
3. **实验。** 第二个 agent 设计受控替换、搜索、原生干预、RelP 归因和家族测量。独立的文本生成器仅根据初步解释编写最大／最小响应测试输入，由宿主机工具执行目标模型计算并评分。
4. **修订。** 第三个 agent 结合初步解释与实测结果，修订解释、批评、不确定性和图注。
5. **发布。** 固定代码生成交互式报告。如果提供本地参考快照，发布工具在研究结束后加入双栏比较。

公开报告默认抽取 **500 个已有位置**，按各激活区间的 log-count 柱高分配数量，再在区间内均匀抽样。Activation spectrum 与 Inside the neuron 复用这些位置及初步标签。修订后，标签仍对应初步解释。具体实现见[方法说明](docs/method.md)、[运行配置](docs/running.md)与[输入数据约定](docs/data.md)。

## 如何评估报告中的证据 <a id="evaluating-the-report-evidence"></a> <a id="how-we-evaluate-an-explanation"></a>

报告提供多种检验解释的方式。目录中的 1–5 分由 LLM 评估与参考解释的语义一致程度。解释是否正确、因果判断是否成立，需要结合以下实验；措辞相似本身不足以证明这些判断。

| 维度 | 问题 | 报告中的证据 |
| --- | --- | --- |
| 覆盖度 | 能否解释正负两极和重要的剩余案例？ | 激活谱、样例、语义聚类、预测错误与 No visible cue 案例。 |
| 特异性 | 能否识别具体触发条件，并与周围主题区分？ | 受控替换、反例、搜索和盲测。 |
| 预测能力 | 能否预测形成解释时未使用的输入？ | 独立的最大／最小响应盲测仅根据解释生成候选文本，受控模板与家族测量提供补充证据。 |
| 因果相关性 | 改变神经元或输入，是否产生预期影响？ | 原生干预、消融、受控输入修改和输出测量。 |
| 稳定性 | 换语料、样本或 agent 运行，解释是否仍成立？ | 跨语料比较，以及计划中的重复运行研究。 |
| 不确定性 | 哪些现象仍未解释，或证据不足以区分？ | 明确的批评、替代解释、弱响应极和失败测试。 |

RelP 使用修改后的导数提供辅助归因证据，仅凭归因图无法确立原生因果效应。具体见[测量范围](docs/method.md#relp-and-intervention-scope)。

## 当前案例中的观察 <a id="observations-from-the-current-comparisons"></a> <a id="what-we-have-found-so-far"></a><a id="key-findings"></a>

### 概念相近，响应条件仍可能不同

下面四个 SlimPajama 100M 案例与原报告有可辨认的共同概念。这些判断来自对个别案例的定性阅读，剩余差异仍需实验检验。

| 神经元 | 两份解释的共同点 | 仍有差异的部分 |
| --- | --- | --- |
| [L50_N3507](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L50_N3507/) | 数学符号的角色区分了正负两极：为对象命名，或作为表达形式中的占位符。 | 具体字母、句式和干预条件。 |
| [L52_N24188](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L52_N24188/) | 残余物与从过去留存至今的事物。 | 是预期即将出现的遗存词，还是响应词本身；较弱的负极。 |
| [L24_N17907](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L24_N17907/) | 被描述为必要的亲身体验，例如“has to be seen”。 | 推荐语和“难忘体验”等表达是否属于同一概念。 |
| [L15_N10957](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L15_N10957/) | 物理防护与加固，例如安全屋和风暴避难所。 | 对复合词中心词的依赖，以及能否推广到屏障或盔甲。 |

另一些案例暴露了更大的差距。例如，我们对 [L31_N23558](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L31_N23558/) 的解释覆盖了宽泛的关系边界，原报告则区分了两类数学表达。这些差异提示我们，需要扩大语料覆盖、设计更能区分候选解释的对照，并加强因果检验。

### 不同语料呈现不同的激活范围

即使都在约 100M tokens 的规模，不同语料也会呈现不同的激活范围。L51_N20229 的原生 product 在 Common Pile 中为 **-19 至 +59**，在 Diverse 中为 **-35 至 +98**，在 Reasoning Mix 中为 **-32.75 至 +111**。这些已观测到的极值不能视为神经元的响应边界。各语料的采样和上下文处理方式也有差异，[数据文档](docs/data.md#corpora-used-in-the-comparisons)列出了具体来源与流程。

相同的 token 预算仍可能覆盖不同的上下文和激活尾部。极值范围更大本身不足以证明解释更好，还需检查正负两极的具体案例和受控测试的响应。

<p align="center">
  <a href="assets/corpus-coverage.svg">
    <img src="assets/corpus-coverage.svg" alt="不同语料中观测到的激活范围" width="90%">
  </a>
</p>

当前报告仍有三处不确定性：采样语料是否包含相关上下文，agent 是否在评分位置看到足够的证据，以及给出的解释能否预测后续测量。需要重复运行并匹配比较条件，才能区分这些影响。

## 安装与使用 <a id="use-neuron-microscope"></a><a id="installation-and-usage"></a><a id="quick-start"></a>

目前可以通过以下三种方式使用项目。

| 方式 | 可以做什么 | 需要什么 |
| --- | --- | --- |
| **浏览结果** | 查看 10 份交互式神经元对比与一次完整研究记录。 | 浏览器。 |
| **体验渲染器** | 生成并查看小型合成报告，用于检查软件与界面功能。 | Python；无需 API 密钥、Docker、GPU 或模型下载。 |
| **开展研究** | 分析已有激活归档中的神经元，发布新报告。 | Linux、Docker、Python 3.10+、CUDA 主机、模型权重、激活数据和 LLM 端点。 |

### 1. 体验离线渲染器

```bash
git clone https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt

python examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

打开 <http://localhost:8000/>。页面使用合成内容，不能作为实验证据。

### 2. 配置完整研究

完整 pipeline 当前需要：

- Linux、Bash 4+、Docker 和 Python 3.10+；
- 支持 CUDA 的主机，以及本地 dense-Qwen3 权重；
- 符合[输入约定](docs/data.md)的激活归档和各神经元的初始报告包；
- 用于研究 agent、辅助打标和文本生成的 LLM 端点。

安装宿主机依赖并构建 agent 镜像：

```bash
python -m pip install -r requirements/gpu.txt
# 安装与 NVIDIA 驱动匹配的 PyTorch CUDA wheel。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

docker build -t neuron-microscope:local .
cp settings.json settings.local.json
```

修改 `settings.local.json` 中的模型、激活归档、初始报告、状态目录和可选参考页面路径，并按可用显卡配置 `gpu_groups`。复现已记录的 LLM 设置时，将 `llm.model` 设为 `deepseek-v4.1-flash-1017`，`llm.reasoning_effort` 设为 `high`，并使用提供该模型的端点。在宿主机环境中提供 `DEEPSEEK_API_KEY`，或通过 `NEURON_PAPER_API_KEY_FILE` 指向本地凭证文件。

然后准备、执行和发布：

```bash
NEURON_PAPER_BUDGET_USD=5 ./run.sh prepare \
  --settings settings.local.json \
  --output runs/example \
  --neurons L55_N11703

./run.sh execute --run runs/example

./run.sh publish \
  --run runs/example \
  --neuron L55_N11703 \
  --output results/example

python -m http.server 8000 --directory results/example
```

将示例神经元 ID 替换为输入数据中存在的编号。批量执行时，解释与修订可以并行；同一 GPU 组的完整实验阶段通过锁依次执行。

完整配置、续跑、单独阶段、测量复用、GPU 分组、预算与发布行为，见[运行说明](docs/running.md)。

## 实测耗时与开销 <a id="recorded-runtime-and-cost"></a> <a id="reproducibility-and-cost"></a>

每次研究都会保留冻结的配置、源码快照、API 请求与响应、token 费用估算、中间交接、实验、测量、修订和最终报告。新的解释仅根据目标激活证据形成，输入中不含旧解释或官方标签。

下面是 **2026 年 10 月 29 日 Colorado／Diverse 100M 研究记录**中的数据：

| 项目 | 记录值 |
| --- | --- |
| 被解释模型 | Qwen3-32B |
| 已有证据 | Diverse 100M；抽取 500 个位置，绘制其中已标注的 495 个位置 |
| 研究与辅助调用 | `deepseek-v4.1-flash-1017`，推理级别 `high` |
| GPU 组 | 4 × NVIDIA A100 40 GB |
| 实测阶段执行时间 | 38 分 00 秒 |
| GPU 实验时间 | 10 分 48 秒，换算为四卡等效时间 |
| API 流式响应时间 | 17 分 22 秒 |
| API 费用估算 | $0.4871 |
| 从开始到完成 | 1 小时 59 分 25 秒，其中包含 1 小时 21 分 25 秒的阶段间隔 |

GPU 时间包含模型加载。GPU 与 API 工作可能重叠，因此两者时间不能直接相加得到阶段执行时间。

以上来自单次实测，无法代表每个神经元的平均开销。费用不含之前的激活采集，并取决于所配置的模型端点和 token 单价。[已发布报告](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/reports/04-published.html#interpretation-costs)提供了底层明细。

## 局限与未来工作 <a id="limitations-and-future-work"></a>

### 当前局限

- 首批对比仅包含 10 个神经元，无法作为对 Qwen3-32B 的随机抽样评估。
- 完整激活归档尚无公开下载入口，外部用户目前需要提供兼容的输入数据。
- 离线演示使用合成内容，不能用于复现已发布的科学结果。
- 语料覆盖、提示词选择、证据展示与 agent 随机性都可能改变解释。
- 目录中的 1–5 分衡量语义一致程度，不能证明机制解释正确。
- 负极和弱激活的解释往往不如强正向响应清楚。
- 我们尚未完成条件匹配的原生神经元与 SAE 比较。

### 未来工作 <a id="future-work"></a>

- [ ] 发布小型真实神经元证据包，让用户无需完整模型或激活归档即可重新生成一份科学报告。
- [ ] 从选定案例扩展到更大规模的随机神经元样本。
- [ ] 衡量解释在不同 agent 运行、提示词、样本与语料间的稳定性。
- [ ] 按统一评估流程，比较条件匹配的原生神经元与 SAE 特征。
- [ ] 逐项检验 AlreadyClean 的主要主张，包括弱激活消融与递归泛化。
- [ ] 支持更多模型、层、语言和文体。
- [ ] 将原型部署地址迁移到稳定的项目或机构域名。

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
| [运行](docs/running.md) | 安装依赖，配置 GPU 与凭证，运行或继续一批实验。 |
| [输入数据](docs/data.md) | 准备激活归档和目标报告包。 |
| [方法与实现](docs/method.md) | 从流程找到代码，了解测量范围。 |
| [报告指南](docs/reports.md) | 查找某张图或某个章节的数据来源和生成阶段。 |
| [开发](docs/development.md) | 运行 CPU 软件检查与离线演示。 |
| [Agent skills](skills/README.md) | 查看解释、实验与修订阶段的指令。 |

## 引用 <a id="citation"></a>

如果在学术工作中使用 Neuron Microscope，请引用本仓库，并注明具体使用的版本或报告：

```bibtex
@software{ruc_trustworthyai_neuron_microscope_2026,
  author = {{RUC-TrustworthyAI}},
  title = {Neuron Microscope: Reproducing and Stress-Testing Native Neuron Explanations},
  year = {2026},
  url = {https://github.com/RUC-TrustworthyAI/Neuron-Microscope}
}
```

## 致谢 <a id="acknowledgements"></a>

报告结构参考 [Goodfire 的神经元报告](https://already-clean.goodfire.com/)。RelP 适配自 [`TransluceAI/circuits`](https://github.com/TransluceAI/circuits)，来源版本记录在 `tools/relp-source.json`。Agent 使用官方 [DeepSeek Harness SDK](https://github.com/deepseek-ai/deepseek-harness/tree/master/python/sdk)。

## 许可 <a id="license"></a>

本仓库代码采用 [MIT License](LICENSE)。第三方组件保留各自原有许可证。
