<p align="center">
  <img src="assets/neuron-microscope.png" width="100%" alt="Neuron Microscope">
</p>

<h1 align="center">
  Neuron Microscope<br>
  <sub>🔬 用 agent 复现 AlreadyClean</sub>
</h1>

<p align="center">
  <a href="docs/method.md"><img src="assets/badges/model.svg" alt="Model: dense Qwen3"></a>
  <a href="skills/README.md"><img src="assets/badges/skills.svg" alt="Agent skills: 3 stages"></a>
  <a href="docs/reports.md"><img src="assets/badges/reports.svg" alt="Reports: interactive HTML"></a>
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh"><img src="assets/badges/live-demo.svg" alt="Showcase：在线展示"></a>
  <a href="#installation-and-usage"><img src="assets/badges/quickstart.svg" alt="Usage: install and run"></a>
</p>

<p align="center"><a href="README.md">English</a> · <b>简体中文</b></p>

<p align="center">
  <a href="#overview">📖 概览</a> ·
  <a href="#reports-and-walkthrough">▶ 结果与流程展示</a> ·
  <a href="#key-findings">🧪 发现与实践经验</a> ·
  <a href="#future-work">🔭 未来工作</a> ·
  <a href="#related-work">📑 我们的相关研究</a> ·
  <a href="#installation-and-usage">🚀 安装与使用</a> ·
  <a href="#documentation">📚 文档</a>
</p>

## 📖 项目概览 <a id="overview"></a>

我们复现了 [AlreadyClean](https://already-clean.goodfire.com/) 的主要神经元分析与报告生成功能，并通过独立实验考察原博客的发现能否复现。

我们研究 **Qwen3-32B** 的原生 MLP 神经元，并公开以下三部分工作：

- **激活证据：** 报告展示选取的上下文，以及 gate、up 和 product 的实测值；[数据文档](docs/data.md)提供归档格式与读取工具。完整激活归档目前尚无公开下载入口。
- **解释 pipeline 及工具：** [三个阶段的 agent skills](skills/README.md)负责理解证据、设计实验与修订解释，并通过[测量和报告工具](docs/method.md)完成实验与展示。
- **交互式可视化：** [神经元报告](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)支持与 AlreadyClean 原报告并排比较；[流程演示](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh)展示一次完整的解释过程。

我们希望读者能够检查证据、复现实验，并共同判断目前能解释多少神经元行为。通过分享成功案例、反例与实践中的困难，大家可以在已有结果上继续改进解释。

## ▶ 结果与流程展示 <a id="reports-and-walkthrough"></a><a id="live-demo"></a><a id="reproduction-results"></a>

### 1. 将一个神经元与原报告并排比较

**[查看 Colorado 对比 →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/)**

点击 **Compare reports** 打开双栏后，**左侧**是原博客公布的报告，**右侧**是我们的独立复现。两者都发现了较强的 Colorado 正向响应。在我们的 SlimPajama 100M 实验中，使用句式 `Officials in {city} approved the annual budget this week.`，在单个 token ` approved` 上评分，四个 Colorado 城市产生 **+22.0 至 +43.3** 的激活，其余六个城市均不超过 **+4.75**。

两份解释仍有差异：原报告提出，gate 会在即将填入地区相关信息的位置打开；我们的解释覆盖了更宽泛的地名与地方新闻上下文。负极的解释也仍不充分。读者可以通过上下文、受控实验和干预结果进一步判断这些差异。

[![Colorado 神经元对比：左侧为原报告，右侧为复现结果](assets/colorado-comparison.jpg)](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/)

### 2. 浏览持续扩充的神经元结果

**[浏览神经元报告 →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)**

当前 SlimPajama 100M 对比集包含 **10 个神经元**，后续会继续扩充。目录展示两份解释，并提供完整双栏报告的入口。其中的 1–5 分是 **LLM 对解释语义相似度的评价**；解释是否成立，需要结合报告里的实验判断。

初步逐项对读后，我们认为以下四个案例在核心概念上比较接近，同时仍有值得检验的差异：

| 神经元 | 两份解释的共同点 | 仍有差异的部分 |
|---|---|---|
| [L50_N3507](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L50_N3507/) | 数学符号的角色区分了正负两极：为对象命名，或作为表达形式中的占位符。 | 具体字母、句式和干预条件。 |
| [L52_N24188](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L52_N24188/) | 残余物与从过去留存至今的事物。 | 是预期即将出现的遗存词，还是响应词本身；较弱的负极。 |
| [L24_N17907](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L24_N17907/) | 被描述为必要的亲身体验，例如“has to be seen”。 | 推荐语和“难忘体验”等表达是否属于同一概念。 |
| [L15_N10957](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L15_N10957/) | 物理防护与加固，例如安全屋和风暴避难所。 | 对复合词中心词的依赖，以及能否推广到屏障或盔甲。 |

另一些案例暴露了更大的差距。例如，我们对 [L31_N23558](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L31_N23558/) 的解释覆盖了宽泛的关系边界，原报告则区分了两类数学表达。这些差异促使我们补充语料覆盖，并设计更能区分候选解释的实验。

[![SlimPajama 100M 神经元对比目录与解释相似度评分](assets/report-directory.jpg)](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)

### 3. 跟随一次完整的解释过程 <a id="workflow"></a>

**[查看流程演示 →](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh)**

演示记录了一个 Colorado 神经元从激活证据到最终报告的过程，依次经历 **解释 → 实验 → 修订**。读者可以查看提示词、工具与实测结果，追踪解释是怎样形成的。该记录使用 [Diverse 100M](docs/data.md#corpora-used-in-the-comparisons)，上面的对比案例使用 SlimPajama 100M。

[![Neuron Microscope 解释流程的运行记录](assets/pipeline-walkthrough-zh.jpg)](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=zh)

![三个研究 agent 依次解释激活证据、在 Qwen3-32B 上开展实验，并修订神经元报告](assets/workflow.png)

*流程示意图中的数字用于说明步骤。实测数值、输入句式与评分位置见上方链接中的报告和流程记录。*

<details>
<summary><strong>解释流程的具体步骤</strong></summary>

1. **准备证据**：从已有激活档案中按幅度区间抽取 token 位置，保留对应上下文，以及实测 gate、up 和乘积激活值。采样参数由 `settings.json` 配置。
2. **初步解释与打标**：第一个独立容器加载 `skills/interpret/`，读取目标 neuron 的初始 HTML 和结构化证据，生成初步解释与分桶规则：正向、负向各最多三组，加一个 No visible cue 组。随后由固定工具根据这些规则逐样本打标，打标时隐藏待评样本的实测响应。代码将标签与实测值对应，用于绘图和修订；Activation spectrum 与 Inside the neuron 中的同一样本复用同一标签。
3. **实验**：第二个独立容器装 experiment skill；agent 设计实验，固定工具在宿主机完成替换、搜索、消融、RelP 归因及其他测量。
4. **修订**：第三个独立容器加载 `skills/revise/`，结合实验测量、初步标签与实测响应，修订解释、Criticism 和图注。固定代码生成 HTML 与费用章节。
5. **发布**：若提供对应原博客的本地快照，发布工具自动加入左右分屏、版块对齐和同步滚动。官方内容只加入展示页，不进入研究 agent 的输入。

仓库默认采用公开报告的采样流程：按各激活区间的 log-count 柱高分配 500 个位置，再在区间内均匀抽样。两张图使用同一批样本及初步解释产生的标签。打标分两路并行，每批最多 256 点；漏返回的样本继续补评，缺项少于 16 点时仅绘制已标注点。具体见[运行配置](docs/running.md#reproduce-the-published-sampling-and-scheduling)和[语料来源](docs/data.md#corpora-used-in-the-comparisons)。

**Clusters** 按语义组织已测量的样本。原始 NMF routes 描述归因模式，一个语义组可以跨越多条 route。各组的样本数、归因贡献占比和上游汇总由实测数据计算。

</details>

## 🧪 发现与实践经验 <a id="key-findings"></a>

### 开销：一次解释需要哪些资源

下面以 **2026 年 10 月 29 日流程演示中的 Colorado／Diverse 100M 实验**为例。[最终报告](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/reports/04-published.html#interpretation-costs)提供了对应的开销明细。

| 设置 | 本次运行配置 |
|---|---|
| 被解释模型 | Qwen3-32B |
| 研究 agent 与辅助 LLM 调用 | deepseek-v4.1-flash-1017，推理级别 `high` |
| GPU 实验组 | 4 × NVIDIA A100 40 GB |
| 已有激活证据 | Diverse 100M；抽取 500 个位置，绘制其中已标注的 495 个位置 |

| 步骤 | GPU 实验时间 | API 流式响应时间 | 阶段执行时间 | API 费用估算 |
|---|---:|---:|---:|---:|
| 整理已有证据 | 0 秒 | 0 秒 | 6 秒 | $0.0000 |
| 解释与打标 | 0 秒 | 6 分 09 秒 | 11 分 46 秒 | $0.1567 |
| 设计与测量实验 | 10 分 48 秒 | 5 分 38 秒 | 18 分 43 秒 | $0.1325 |
| 修订与审读 | 0 秒 | 5 分 36 秒 | 7 分 03 秒 | $0.1980 |
| 渲染报告 | 0 秒 | 0 秒 | 23 秒 | $0.0000 |
| **合计** | **10 分 48 秒** | **17 分 22 秒** | **38 分 00 秒** | **$0.4871** |

从开始到完成共经过 **1 小时 59 分 25 秒**，其中包含 **1 小时 21 分 25 秒的阶段间隔**。GPU 时间包含模型加载，并换算为四卡等效时间。API 响应与其他工作可能重叠，因此 GPU 时间和 API 时间不能直接相加得到执行时间。表内时间和费用分别四舍五入。费用按本次配置的 token 单价估算，不含此前的激活采集；这些数字来自单个案例，不能作为每个神经元的平均开销。

GPU 工作集中在实验阶段，解释、打标和修订也会消耗 LLM 调用与 CPU 时间。批量运行时，解释和修订可以与实验并行；同一 GPU 组通过锁保证每次只有一个实验阶段使用它。

### 语料覆盖会改变可见的证据

即使都在约 **100M tokens** 的规模，不同语料也会呈现不同的激活范围。例如，**L51_N20229** 的原生 product 在 Common Pile 中为 **-19 至 +59**，在 Diverse 中为 **-35 至 +98**，在 Reasoning Mix 中为 **-32.75 至 +111**。下图同时列出了 SlimPajama 和 FineWeb 的结果。[数据文档](docs/data.md#corpora-used-in-the-comparisons)列出了各语料的来源、抽样方式及 Diverse 配比。

![L51_N20229 在五套约 100M-token 语料中的原生激活最小值与最大值](assets/corpus-coverage.svg)

这些是已观测到的极值，不能视为神经元的响应边界。更大的范围本身也不足以证明解释更好。实践中需要检查正负两极的具体文本，并在更多上下文里检验解释；仅凭 token 数量，无法判断覆盖是否充分。

### 需要继续分析的三类不确定性

1. **语料能否激发相关行为？** 某些语言、文体或句式的样本可能不足。在判断神经元难以解释之前，需要先比较语料覆盖和正负激活尾部。
2. **Agent 能否清楚地看到证据？** 前缀长度、评分位置、token 染色和样本选择都会影响判断。保留实测上下文，并让两张图复用同一批标注位置，有助于读者追溯判断依据。
3. **Agent 能否形成并检验合适的解释？** 宽泛主题可能符合样例，却遗漏具体触发条件，Colorado 的对比就体现了这一点。受控替换、反例和盲测可以帮助区分候选解释；解释的稳定性还需要通过重复运行衡量。

**关于 superposition 与 recursive generalization。** 初步阅读这些案例时，我们的主观感受是，部分原生神经元的解释与此前研究的 SAE 特征相当清楚，个别案例更清晰。目前观察数量较少，尚不足以得出普遍优势的结论。接下来需要在匹配条件下比较 SAE，并检验表面不同的行为是否共享机制，才能进一步判断叠加与递归泛化。

## 🔭 未来工作 <a id="future-work"></a>

欢迎从以下三个方向参与：

- **数据：** 补充语言和文体覆盖，分享难以解释的上下文，并整理来源与再分发条件清楚、便于复用的激活数据集。
- **解释与训练：** 改进解释方法，研究训练如何影响特征的形成，并用实测结果和反例检验改进。
- **Pipeline 不确定性：** 衡量采样、证据展示、提示词与 agent 重复运行带来的影响，再针对主要来源解决问题。

可以通过 [GitHub Issues](https://github.com/RUC-TrustworthyAI/Neuron-Microscope/issues) 分享具体案例、实验或建议。附上神经元 ID、测试上下文与观察到的行为，会更方便其他人复现和跟进。

## 📑 我们的相关研究 <a id="related-work"></a>

团队此前的工作涵盖特征学习、概念分析、解释评估与语言模型安全。发表信息也可参阅[王希廷老师的学院主页](https://ai.ruc.edu.cn/szdw/zrjs/wxt/index.htm)。

- **C²R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders** — Haoran Jin 等，**ICML 2026**。通过跨样本的一致性约束，缓解 SAE 中的特征分裂与吸收。[论文](https://arxiv.org/abs/2606.30609) · [代码](https://github.com/hr-jin/Cross-sample-Consistency-Regularization)。
- **Uncovering Safety Risks of Large Language Models through Concept Activation Vector** — Zhihao Xu 等，**NeurIPS 2024**。利用安全概念激活向量，研究语言模型的安全机制及其脆弱性。[论文](https://arxiv.org/abs/2404.12038) · [代码](https://github.com/SproutNan/AI-Safety_SCAV)。
- **Evaluating Readability and Faithfulness of Concept-based Explanations** — Meng Li 等，**EMNLP 2024**。形式化概念解释，提出可读性、忠实度的度量方法，并评估这些度量本身。[论文](https://aclanthology.org/2024.emnlp-main.36/) · [代码](https://github.com/hr-jin/Concept-Explanation-Evaluation)。
- **Do LLMs Know Their Vulnerable Scenarios?** — Ziheng Peng 等，**arXiv 2026**。利用 SAE 概念与因果归因，识别会削弱拒答的场景，并分析场景概念之间的交互。[论文](https://arxiv.org/abs/2607.23496)。

## 🚀 安装与使用 <a id="installation-and-usage"></a><a id="quick-start"></a>

### 1. 在本地体验报告渲染

```bash
git clone https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt
python examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

打开 **http://localhost:8000/** 查看演示。

演示使用代码生成的合成数据，仅检查页面功能，不是实验结果。它不需要 API、Docker 或模型，也可以在 macOS 上运行。

### 2. 配置真实实验

完整流程需要 Linux、Bash 4+、Docker、Python 3.10+、本地 dense-Qwen3 权重、已有激活和初始报告包，以及 DeepSeek API 凭证。复现这里的运行配置时，将 `llm.model` 设为 `deepseek-v4.1-flash-1017`、`llm.reasoning_effort` 设为 `high`，并使用提供该模型的 API 端点。GPU 只承担目标模型的测量，四卡共同承载一份模型。

```bash
python -m pip install -r requirements/gpu.txt
# 安装与 NVIDIA 驱动匹配的 CUDA wheel。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
docker build -t neuron-microscope:local .
cp settings.json settings.local.json
```

修改 `settings.local.json` 中的 `paths.reports`、`paths.activations`、`paths.model`、`paths.state` 和 `gpu_groups`。可选 `paths.references` 放 `Lx_Ny.html` 原博客快照。相对路径以配置文件所在目录为基准。报告包及激活格式见 [docs/data.md](docs/data.md)。

在宿主机环境中提供 `DEEPSEEK_API_KEY`，或通过 `NEURON_PAPER_API_KEY_FILE` 指向本地密钥文件。不要把密钥写进配置或 Git。

### 3. 运行并查看

```bash
NEURON_PAPER_BUDGET_USD=5 ./run.sh prepare \
  --settings settings.local.json --output runs/example --neurons L55_N11703
./run.sh execute --run runs/example
./run.sh publish --run runs/example --neuron L55_N11703 --output results/example
python -m http.server 8000 --directory results/example
```

打开 **http://localhost:8000/**。示例 neuron 需替换为自己的数据中存在的编号。

- `prepare`：冻结代码和配置，不启动 LLM 或模型实验。
- `execute`：按快照执行准备和三阶段流程。
- `publish`：保留生成的科学内容，加入统一显示与可选对比功能。
- 默认整批 API 上限 2 美元；示例设为 5 美元。设 `unlimited` 可取消上限。费用是配置单价计算的估算，不是账户账单。
- 批量运行时，在 `--neurons` 后列出多个神经元。默认允许不同神经元的解释与修订并行；同一 GPU 组的完整实验阶段通过阶段锁依次执行。多组 GPU 并行时，每组准备一批任务，共用 `paths.state`，具体见[双组配置](docs/running.md#reproduce-the-published-sampling-and-scheduling)。续跑、独立阶段和测量复用见[运行文档](docs/running.md#start-continue-and-publish)。
- 完整请求、输出、测量、费用和耗时保存在 `runs/`，可按 neuron 和阶段查看。

更多续跑、单阶段和复用测量操作见 [运行说明](docs/running.md)。展示窗口只影响页面文本：

- **Exemplars 与 Clusters**：默认折叠预览显示圈定 token 及其之前最多32个 token；点击 **Show full context** 展开保存的上下文。
- **Activation spectrum**：从首个高亮激活前最多128个 token 开始展示，并保留已有后文。若没有 token 达到高亮阈值，则以圈定位置为锚点。

测量使用的输入前缀和评分位置保持不变。

## 📚 文档导航 <a id="documentation"></a>

| 文档 | 解决什么问题 |
|---|---|
| [运行说明](docs/running.md) | 安装依赖，配置 GPU、认证，启动或继续一批实验 |
| [输入数据](docs/data.md) | 准备激活档案和目标报告数据 |
| [方法与实现](docs/method.md) | 从流程找到代码，了解各类测量的口径 |
| [报告指南](docs/reports.md) | 查找某张图或某个章节的数据来源和生成阶段 |
| [开发说明](docs/development.md) | 运行 CPU 软件测试与离线演示 |

<details>
<summary><strong>仓库目录</strong></summary>

```text
run.sh                 容器启动、排队与阶段交接
settings.json          科学参数默认值及可配置的宿主路径
Dockerfile             Agent 镜像与固定版本的 harness
skills/
  interpret/           初步解释
  experiment/          实验设计与工具调用
  revise/              修订、Criticism 和图注
tools/                 宿主证据读取、测量、渲染与费用记录
analysis/              从已有激活生成初始报告和统计表
examples/demo.py       离线合成报告示例
assets/                项目头图、流程图和配图
tests/                 CPU 软件样例和微型随机模型测试
requirements/          CPU、GPU 宿主及测试依赖
docs/                  环境、数据、方法与报告指南
```

</details>

## 🙏 致谢 <a id="acknowledgements"></a>

报告结构参考 Goodfire 的 [Neural Networks Are Already Clean](https://already-clean.goodfire.com/)，RelP 适配参考 [TransluceAI/circuits](https://github.com/TransluceAI/circuits)，agent 执行使用 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)。

## 📄 许可 <a id="license"></a>

项目原创代码使用 [MIT License](LICENSE)。
