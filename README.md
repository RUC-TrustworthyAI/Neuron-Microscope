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
  <a href="https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en"><img src="assets/badges/live-demo.svg" alt="Live Demo: online"></a>
  <a href="#quick-start"><img src="assets/badges/quickstart.svg" alt="Quick start: offline demo"></a>
</p>

<p align="center"><b>English</b> · <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="#overview">📖 Overview</a> ·
  <a href="#live-demo">▶ Live Demo</a> ·
  <a href="#workflow">⚗️ How It Works</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="#reproduction-results">🧪 Reproduction Results</a> ·
  <a href="#documentation">📚 Documentation</a>
</p>

## 📖 Overview <a id="overview"></a>

We reproduce the main neuron analysis and report-generation features of [AlreadyClean](https://already-clean.goodfire.com/) and independently investigate whether the original blog's findings can be reproduced. Our experiments show **[KEY FINDINGS]** and lead us to **[REPRODUCTION CONCLUSIONS]**. The supporting experiments and analysis are presented in [Reproduction Results](#reproduction-results).

We also open-source the complete three-stage pipeline for investigating native MLP neurons in dense Qwen3 models. It covers evidence preparation, initial interpretation, experiment design and execution, explanation revision, and interactive HTML reporting. We add **[ADDITIONAL FEATURES]** for the community to use and build on.

Next, we plan to **[FUTURE PLANS]**. We welcome test results and feedback to help improve the neuron interpretation workflow together.

## ▶ Live Demo <a id="live-demo"></a>

**[Browse neuron reports →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/)** · **[Explore the pipeline walkthrough →](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en)**

Browse interactive neuron reports, with side-by-side reference comparisons where available. The bilingual walkthrough follows a recorded Colorado neuron investigation through evidence preparation and all three agent stages, including prompt examples, measured experiments and saved reports. Both websites are publicly accessible.

[![Preview of the Neuron Microscope walkthrough](assets/live-demo-en-hd.png)](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en)

See [the demo guide](docs/live-demo.md) for local preview and recording details.

## ⚗️ How It Works <a id="workflow"></a>

![Evidence preparation and the three agent containers, with JSON/HTML handoffs and model measurements performed by host tools](assets/workflow.png)

1. **Prepare evidence.** Sample token positions across activation intervals from the saved archive and retain their contexts and measured gate, up, and product values. Sampling parameters are set in `settings.json`.
2. **Interpret and label.** A fresh agent container loads `skills/interpret/` and reads the target neuron's initial HTML and structured evidence. It writes an initial explanation and response rules: up to three positive groups, up to three negative groups, and one No visible cue group. Fixed tools then use those rules to label individual contexts without revealing their measured responses. The labels are joined to the measurements for visualization and revision; shared positions reuse the same label across Activation spectrum and Inside the neuron.
3. **Experiment.** A fresh container loads `skills/experiment/`. Its tools run replacement sweeps, searches, native interventions, RelP attribution, family measurements, and blind maximum/minimum tests on the host. Saved input prefixes and scoring positions are retained.
4. **Revise.** A fresh container loads `skills/revise/`, reviews the experiments and the initial labels alongside measured responses, and revises the explanation, Criticism, and captions. Fixed code assembles the report and cost records.
5. **Publish.** Export the generated report. If a local reference snapshot is available, publication adds a collapsible side-by-side comparison with aligned sections and linked scrolling. Reference content is added only at publication; it is not given to the research agents.

The published reports use a 500-position sampling configuration. The repository default is currently up to 256 scan points; use the configuration saved with a report when reproducing it.

Each agent runs in a separate container with one skill. Host tools handle paths, GPU allocation, accounting, and HTML generation. The default research agent is `deepseek-flash` with `reasoning_effort: high`.

**Semantic Clusters** group measured examples by meaning. Raw NMF routes describe attribution patterns, and a semantic cluster may span several routes. Counts, attribution-mass shares and upstream summaries are computed from the measured members.

## 🚀 Quick Start <a id="quick-start"></a>

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

This is an excerpt: keep the other settings in the copied file. Relative paths resolve against that file's directory. The report directory contains `Lx_Ny/` packages; reference snapshots, when used, are named `Lx_Ny.html`.

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

For a batch, list several neurons after `--neurons`. By default, the next neuron starts after the previous experiment stage has completed; revision may overlap the next interpretation. `gpu_groups` can list independent GPU groups. See [continuation, isolated stages, and saved-measurement reuse](docs/running.md#start-continue-and-publish).

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

## 🧪 Reproduction Results <a id="reproduction-results"></a>

[MAIN COMPARISON FIGURES: ORIGINAL BLOG VS. OUR REPRODUCTION]

- **Findings:** [KEY FINDINGS AND SUPPORTING EXPERIMENTS]
- **Reproduction conclusions:** [WHICH ORIGINAL FINDINGS WERE REPRODUCED, PARTIALLY REPRODUCED, OR NOT REPRODUCED]

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
