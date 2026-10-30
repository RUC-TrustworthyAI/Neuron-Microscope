<p align="center">
  <img src="assets/neuron-microscope.png" width="100%" alt="Neuron Microscope">
</p>

<h1 align="center">
  Neuron Microscope<br>
  <sub>🔬 Reproducing AlreadyClean with agents</sub>
</h1>

<p align="center">
  <a href="docs/method.md"><img src="assets/badges/model.svg" alt="Model: dense Qwen3"></a>
  <a href="skills/README.md"><img src="assets/badges/skills.svg" alt="Agent skills: 3 stages"></a>
  <a href="docs/reports.md"><img src="assets/badges/reports.svg" alt="Reports: interactive HTML"></a>
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en"><img src="assets/badges/live-demo.svg" alt="Showcase: online"></a>
  <a href="#installation-and-usage"><img src="assets/badges/quickstart.svg" alt="Usage: install and run"></a>
</p>

<p align="center"><b>English</b> · <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="#overview">📖 Overview</a> ·
  <a href="#reports-and-walkthrough">▶ Explore the Results</a> ·
  <a href="#key-findings">🧪 Findings &amp; Lessons</a> ·
  <a href="#future-work">🔭 Future Work</a> ·
  <a href="#related-work">📑 Our Related Work</a> ·
  <a href="#installation-and-usage">🚀 Installation &amp; Usage</a> ·
  <a href="#documentation">📚 Documentation</a>
</p>

## 📖 Overview <a id="overview"></a>

We reproduce the main neuron analysis and report-generation features of [AlreadyClean](https://already-clean.goodfire.com/) and independently investigate whether the original blog's findings can be reproduced.

Our investigations focus on native MLP neurons in **Qwen3-32B**. We share three parts of the work:

- **Activation evidence:** sampled contexts and measured gate, up, and product values in the reports, with [data formats and reading tools](docs/data.md). Full activation archives do not yet have a public download.
- **The explanation pipeline and tools:** [three agent skills](skills/README.md) for interpreting evidence, designing experiments, and revising explanations, backed by [measurement and reporting tools](docs/method.md).
- **Interactive visualization:** [neuron reports](https://neuron-microscope-papers.tangysiren10.chatgpt.site/) that put our results beside AlreadyClean's published reports, and a [recorded pipeline walkthrough](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en).

We want readers to inspect the evidence, reproduce the process, and help establish how much of a neuron's behavior we can explain. Sharing successful tests, counterexamples, and practical difficulties should make it easier to improve these explanations together.

## ▶ Explore the Results <a id="reports-and-walkthrough"></a><a id="live-demo"></a><a id="reproduction-results"></a>

### 1. Compare one neuron with the original report

**[Open the Colorado comparison →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/)**

Select **Compare reports** to open the two panes. The original report is on the **left**, and our independent reproduction is on the **right**. Both identify a strong Colorado-related positive response. In our SlimPajama 100M run, using `Officials in {city} approved the annual budget this week.` and scoring the single token ` approved` gives **+22.0 to +43.3** for four Colorado cities, while the six other cities stay at **+4.75 or below**.

The agreement is partial: the original explains a gate that opens before a place-specific fill-in value, while our report gives a broader place-and-local-news account. The negative pole remains less clearly explained. The comparison lets readers examine that difference through contexts, controlled tests, and interventions.

[![Colorado neuron: original report on the left and reproduction on the right](assets/colorado-comparison.jpg)](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L55_N11703/)

### 2. Browse the growing set of neuron comparisons

**[Browse neuron reports →](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)**

The current SlimPajama 100M comparison set contains **10 neurons**, with further investigations planned. Each directory row links to the full side-by-side report and lists both interpretations. The 1–5 score is an **LLM assessment of semantic similarity**; the experiments in each report provide the evidence for its claims.

Our preliminary reading finds close agreement on the core concept in these four cases, with remaining differences worth testing:

| Neuron | Shared interpretation | What still differs |
|---|---|---|
| [L50_N3507](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L50_N3507/) | A mathematical symbol's role separates the two poles: naming an object versus occupying an expression form. | Exact letter, construction, and steering conditions. |
| [L52_N24188](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L52_N24188/) | Remnants and things surviving from the past. | Anticipating a remnant word versus responding to the word itself; the weak negative pole. |
| [L24_N17907](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L24_N17907/) | Firsthand experience presented as necessary, such as “has to be seen.” | How broadly recommendations and unforgettable-experience phrases belong to the concept. |
| [L15_N10957](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L15_N10957/) | Physical protection and fortification, including safe rooms and storm shelters. | Compound-head specificity and generalization to barriers or armor. |

Other cases expose larger gaps. For example, [L31_N23558](https://neuron-microscope-papers.tangysiren10.chatgpt.site/L31_N23558/) receives a broad relational-boundary explanation in our run, whereas the original distinguishes two mathematical frames. These cases motivate better corpus coverage and more discriminating experiments.

[![Directory of SlimPajama 100M neuron comparisons and explanation similarity scores](assets/report-directory.jpg)](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)

### 3. Follow a complete investigation <a id="workflow"></a>

**[Explore the pipeline walkthrough →](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en)**

Follow a recorded Colorado investigation from the activation evidence through the three agent stages: **interpret → experiment → revise**. The walkthrough shows the prompts, tools, measured experiments, and final report, so readers can trace how the explanation was reached. This recording uses [Diverse 100M](docs/data.md#corpora-used-in-the-comparisons); the comparison above uses SlimPajama 100M.

[![Recorded Neuron Microscope pipeline walkthrough](assets/pipeline-walkthrough-en.jpg)](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en)

![Three research agents interpret activation evidence, run experiments on Qwen3-32B, and revise the neuron report](assets/workflow.png)

*Workflow schematic with illustrative numbers. For measured values, input templates and scoring positions, open the linked reports and walkthrough.*

<details>
<summary><strong>How the investigation runs</strong></summary>

1. **Prepare evidence.** Sample token positions across activation intervals from the saved archive and retain their contexts and measured gate, up, and product values. Sampling parameters are set in `settings.json`.
2. **Interpret and label.** A fresh agent container loads `skills/interpret/` and reads the target neuron's initial HTML and structured evidence. It writes an initial explanation and response rules: up to three positive groups, up to three negative groups, and one No visible cue group. Fixed tools then use those rules to label individual contexts without revealing their measured responses. The labels are joined to the measurements for visualization and revision; shared positions reuse the same label across Activation spectrum and Inside the neuron.
3. **Experiment.** A fresh container loads `skills/experiment/`. Its tools run replacement sweeps, searches, native interventions, RelP attribution, family measurements, and blind maximum/minimum tests on the host. Saved input prefixes and scoring positions are retained.
4. **Revise.** A fresh container loads `skills/revise/`, reviews the experiments and the initial labels alongside measured responses, and revises the explanation, Criticism, and captions. Fixed code assembles the report and cost records.
5. **Publish.** Export the generated report. If a local reference snapshot is available, publication adds a collapsible side-by-side comparison with aligned sections and linked scrolling. Reference content is added only at publication; it is not given to the research agents.

The default matches the published sampling procedure: 500 saved positions, allocated across activation bins in proportion to their log-count heights, with uniform sampling within each bin. Both plots use this same pool and its initial labels. Labeling uses two concurrent batches of up to 256; missing answers are requested again until fewer than 16 points remain, and only labeled points are plotted. See the [run configuration](docs/running.md#reproduce-the-published-sampling-and-scheduling) and [corpus sources](docs/data.md#corpora-used-in-the-comparisons).

Each agent runs in a separate container with one skill. Host tools handle paths, GPU allocation, accounting, and HTML generation. The recorded investigations use `deepseek-v4.1-flash-1017` with `reasoning_effort: high`.

**Semantic Clusters** group measured examples by meaning. Raw NMF routes describe attribution patterns, and a semantic cluster may span several routes. Counts, attribution-mass shares and upstream summaries are computed from the measured members.

</details>

## 🧪 Findings & Practical Lessons <a id="key-findings"></a>

### Cost: what one investigation required

The following example is the **Colorado / Diverse 100M run recorded in the walkthrough on 29 October 2026**. Its [final report](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/reports/04-published.html#interpretation-costs) contains the underlying cost breakdown.

| Setting | Recorded configuration |
|---|---|
| Target model | Qwen3-32B |
| Research and auxiliary LLM calls | deepseek-v4.1-flash-1017, reasoning effort `high` |
| GPU experiment group | 4 × NVIDIA A100 40 GB |
| Saved evidence | Diverse 100M; 500 sampled positions, 495 labeled positions plotted |

| Step | GPU experiment time | API streaming time | Stage execution time | API estimate |
|---|---:|---:|---:|---:|
| Prepare saved evidence | 0 s | 0 s | 6 s | $0.0000 |
| Interpret and label | 0 s | 6 min 09 s | 11 min 46 s | $0.1567 |
| Design and measure experiments | 10 min 48 s | 5 min 38 s | 18 min 43 s | $0.1325 |
| Revise and review | 0 s | 5 min 36 s | 7 min 03 s | $0.1980 |
| Render the report | 0 s | 0 s | 23 s | $0.0000 |
| **Total** | **10 min 48 s** | **17 min 22 s** | **38 min 00 s** | **$0.4871** |

The elapsed time from start to finish was **1 h 59 min 25 s**, including **1 h 21 min 25 s of gaps between recorded stages**. GPU time includes model loading and is expressed as four-GPU equivalent time. API streams and other work can overlap, so GPU and API times do not add up to execution time. Displayed durations and fees are rounded independently. Costs use the run's configured token prices, exclude the earlier activation collection, and are API estimates; this is one measured case, not a per-neuron average.

The experimental stage accounts for the GPU work, while interpretation, labeling, and revision also consume LLM calls and CPU time. For batches, parallel interpretation and revision can overlap with experiments, while a GPU lock lets only one experiment stage use each GPU group at a time.

### Corpus coverage changes the evidence

At approximately the same **100M-token** scale, different corpora expose different activation ranges. For **L51_N20229**, the recorded native product spans **−19 to +59** in Common Pile, **−35 to +98** in Diverse, and **−32.75 to +111** in Reasoning Mix. The figure includes SlimPajama and FineWeb as well. [Corpus sources, sampling and the Diverse mixture](docs/data.md#corpora-used-in-the-comparisons) define these names and token counts.

![Observed minimum and maximum native activation for L51_N20229 across five approximately 100M-token corpora](assets/corpus-coverage.svg)

These are observed extrema, not bounds on what the neuron can do. A larger range alone does not establish a better explanation. The practical lesson is to inspect the text at both poles and test the proposed rule on additional contexts; token count alone does not establish coverage.

### Three sources of uncertainty to investigate

1. **Does the corpus elicit the behavior?** A corpus may contain too few examples of a particular language, genre, or construction. Compare coverage and both activation tails before concluding that a neuron is difficult to interpret.
2. **Does the agent see the evidence clearly?** Prefix length, scoring position, token coloring, and sample selection affect what can be inferred. Retain the measured context and use the same labeled positions across plots so readers can trace a judgment back to its evidence.
3. **Can the agent form and test the right explanation?** A broad topic can fit examples while missing the trigger. The Colorado comparison illustrates this gap. Controlled replacements, counterexamples, and blind text tests help separate competing accounts, and repeated runs are needed to measure explanation stability.

**Superposition and recursive generalization.** Our initial qualitative impression is that some native-neuron explanations are as clear as, or clearer than, SAE features we have studied. The small set examined so far does not establish a general advantage. We still need matched SAE comparisons and tests of whether apparently different behaviors share a mechanism before drawing conclusions about superposition or recursive generalization.

## 🔭 Future Work <a id="future-work"></a>

We welcome contributions in three directions:

- **Data:** broaden language and genre coverage, share difficult contexts, and prepare reusable activation datasets with clear provenance and redistribution terms.
- **Explanation and training:** improve explanation methods and investigate how training affects the features that emerge, using measurements and counterexamples to judge progress.
- **Pipeline uncertainty:** measure the effects of sampling, evidence presentation, prompts, and repeated agent runs, then address the sources that matter most.

Please share a concrete example, experiment, or proposal through [GitHub Issues](https://github.com/RUC-TrustworthyAI/Neuron-Microscope/issues). A neuron ID, the tested context, and the observed behavior help others reproduce and investigate it.

## 📑 Our Related Work <a id="related-work"></a>

Our team's related work spans feature learning, concept-based analysis, explanation evaluation, and interpretable forecasting. Publication details are also listed on [Xiting Wang's faculty page](https://ai.ruc.edu.cn/szdw/zrjs/wxt/index.htm).

- **C²R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders** — Haoran Jin et al., **ICML 2026**. Encourages consistent latent assignments across samples to reduce feature splitting and absorption in SAEs. [Paper](https://arxiv.org/abs/2606.30609) · [Code](https://github.com/hr-jin/Cross-sample-Consistency-Regularization).
- **Uncovering Safety Risks of Large Language Models through Concept Activation Vector** — Zhihao Xu et al., **NeurIPS 2024**. Uses safety concept activation vectors to investigate model safety mechanisms and their vulnerabilities. [Paper](https://arxiv.org/abs/2404.12038) · [Code](https://github.com/SproutNan/AI-Safety_SCAV).
- **Evaluating Readability and Faithfulness of Concept-based Explanations** — Meng Li et al., **EMNLP 2024**. Formalizes concept-based explanations and develops measures of readability and faithfulness, together with a method for evaluating those measures. [Paper](https://aclanthology.org/2024.emnlp-main.36/) · [Code](https://github.com/hr-jin/Concept-Explanation-Evaluation).
- **ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting** — Ziheng Peng et al., **ICLR 2026**. Organizes temporal prototypes hierarchically to explain global patterns and local variations in forecasts. [Paper](https://arxiv.org/abs/2509.23159) · [Code](https://github.com/SKURA502/ProtoTS).

## 🚀 Installation & Usage <a id="installation-and-usage"></a><a id="quick-start"></a>

### 1. Try the report renderer locally

The demo generates a small synthetic fixture; it is not an experimental result. It needs no API key, Docker, GPU, or downloaded model and also works on macOS.

```bash
git clone https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt
python examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

Open **http://localhost:8000/** to view the demo.

For CPU development checks, see [docs/development.md](docs/development.md).

### 2. Configure a real investigation

- Linux, Bash 4+, Docker, and Python 3.10+ for the full pipeline.
- A CUDA-capable host with local dense-Qwen3 model weights. The default GPU group is `[0,1,2,3]`; a group holds one model sharded across its GPUs. The research setup used four A100 40 GB GPUs, but memory use depends on model and prefix length.
- An existing activation archive and per-neuron initial report packages; see [the input contract](docs/data.md).
- A DeepSeek API credential on the host for agents and auxiliary labeling/writing calls.

The agent container does not need CUDA. Model computation runs through host tools, while container inputs are mounted read-only.

```bash
python -m pip install -r requirements/gpu.txt
# Choose a CUDA wheel compatible with your NVIDIA driver:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
docker build -t neuron-microscope:local .
cp settings.json settings.local.json
```

Edit `settings.local.json` to point to your inputs:

```json
{
  "paths": {
    "reports": "/path/to/initial-reports",
    "activations": "/path/to/activation-archive",
    "model": "/path/to/Qwen3-32B",
    "state": "/tmp/neuron-microscope",
    "references": "/path/to/optional-reference-pages"
  },
  "gpu_groups": [[0, 1, 2, 3]]
}
```

This is an excerpt: keep the other settings in the copied file. To match the recorded investigations, set `llm.model` to `deepseek-v4.1-flash-1017` and `llm.reasoning_effort` to `high` for an endpoint that provides this model. Relative paths resolve against that file's directory. The report directory contains `Lx_Ny/` packages; reference snapshots, when used, are named `Lx_Ny.html`.

Supply `DEEPSEEK_API_KEY` through your host environment, or set `NEURON_PAPER_API_KEY_FILE` to a local credential file. Do not put credentials in settings. The agent receives a temporary credential for the host API meter, not the upstream key.

### 3. Run and publish

```bash
NEURON_PAPER_BUDGET_USD=5 ./run.sh prepare \
  --settings settings.local.json --output runs/example --neurons L55_N11703
./run.sh execute --run runs/example
./run.sh publish --run runs/example --neuron L55_N11703 --output results/example
python -m http.server 8000 --directory results/example
```

Open **http://localhost:8000/**. Replace the example neuron with one present in your archive.

`prepare` freezes source and settings without calling an LLM or running the model. `execute` performs evidence preparation and the three stages using that snapshot. `publish` adds presentation controls while keeping the generated scientific content unchanged.

Display windows affect only the rendered text:

- **Exemplars and Clusters:** the collapsed preview shows the marked token and up to 32 preceding tokens. **Show full context** expands the saved context.
- **Activation spectrum:** text starts up to 128 tokens before the first highlighted activation and retains the saved following text. If no token meets the highlighting threshold, the marked position is used as the anchor.

The measured input prefix and scoring position remain unchanged.

The budget applies to all API calls in that batch and stays in host accounting. The default is USD 2; set `NEURON_PAPER_BUDGET_USD=unlimited` explicitly for no cap. Configure the token prices in settings for your API endpoint. They are not account invoices.

For a batch, list several neurons after `--neurons`. By default, interpretation and revision run concurrently across neurons. A stage-level lock serializes the complete experiment stage for the configured GPU group. To use independent groups in parallel, prepare one batch per group with a shared `paths.state`; see the [two-group configuration](docs/running.md#reproduce-the-published-sampling-and-scheduling). See [continuation, isolated stages, and saved-measurement reuse](docs/running.md#start-continue-and-publish).

### 4. Inspect outputs

```text
runs/example/
  manifest.json                 Frozen settings, inputs and budget
  framework-source/             Source snapshot for this run
  logs/api/                     Complete API requests, responses and usage
  logs/usage.json               Accumulated token-price estimates
  units/Lx_Ny/
    01-interpretation/          Preparation and initial explanation
    02-experiments/             Experiments, traces and measurements
    03-revision/                Revised report and independent reading
    */handoff/                  JSON, HTML and artifacts for the next stage
    PIPELINE_TERMINAL.json      Final state and report location
```

Each run retains its requests, measurements and reports. Fresh interpretations use the target activation evidence without previous explanations or official labels.

## 📚 Documentation <a id="documentation"></a>

| Guide | Start here when you want to… |
|---|---|
| [Running](docs/running.md) | Install dependencies, configure GPUs and credentials, run or continue a batch |
| [Input data](docs/data.md) | Prepare an activation archive and target report package |
| [Method and implementation](docs/method.md) | Trace the workflow to code and understand measurement scope |
| [Report guide](docs/reports.md) | Find the source and generation stage of a particular plot or section |
| [Development](docs/development.md) | Run CPU software checks and the offline demo |

<details>
<summary><strong>Repository layout</strong></summary>

```text
run.sh                  Container orchestration and serial handoffs
settings.json           Scientific settings and configurable host paths
Dockerfile              Agent image with pinned harness, jq and ripgrep
skills/                 interpret, experiment, revise
tools/                  Evidence, GPU measurements, rendering and accounting
analysis/               Initial report builders and data-free HTML/JS templates
assets/                 Project banner, workflow diagram and figures
requirements/           Host and development dependencies
tests/                  Software tests and synthetic fixtures
examples/demo.py        Synthetic offline renderer example
docs/                   Setup, data contracts, methods and report guide
```

</details>

## 🙏 Acknowledgements <a id="acknowledgements"></a>

The report structure follows [Goodfire's neuron reports](https://already-clean.goodfire.com/). RelP adapts [TransluceAI/circuits](https://github.com/TransluceAI/circuits), with the source revision recorded in `tools/relp-source.json`. Agents use the official [DeepSeek Harness SDK](https://github.com/deepseek-ai/deepseek-harness/tree/master/python/sdk).

## 📄 License <a id="license"></a>

Original project code uses the [MIT License](LICENSE).
