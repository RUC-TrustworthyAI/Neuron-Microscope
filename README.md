<p align="center">
  <img src="assets/hero.png" width="100%" alt="Neuron Investigator">
</p>

<h1 align="center">
  Neuron Investigator<br>
  <sub>🔬 Investigating neurons with agents and experiments</sub>
</h1>

<p align="center">
  <a href="docs/method.md"><img src="assets/badges/model.svg" alt="Model: dense Qwen3"></a>
  <a href="skills/README.md"><img src="assets/badges/skills.svg" alt="Agent skills: 3 stages"></a>
  <a href="#reports"><img src="assets/badges/reports.svg" alt="Reports: interactive HTML"></a>
  <a href="#quick-start"><img src="assets/badges/quickstart.svg" alt="Quick start: offline demo"></a>
</p>

<p align="center"><b>English</b> · <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="#overview">📖 Overview</a> ·
  <a href="#workflow">⚗️ Workflow</a> ·
  <a href="#reports">🖼️ Reports</a> ·
  <a href="#quick-start">🚀 Quick start</a> ·
  <a href="#documentation">📚 Documentation</a>
</p>

## 📖 Overview <a id="overview"></a>

A three-stage agent pipeline for investigating native MLP neurons in dense Qwen3 models. It starts with saved activation evidence, runs controlled measurements, and produces an interactive HTML report.

This repository contains source code, three skills, development tests, and data-free report templates. Model weights, corpus activations, full generated reports, reference-page snapshots, credentials, and API logs are supplied or generated locally and are not distributed here.

## ⚗️ How It Works <a id="workflow"></a>

![Evidence preparation and the three agent containers, with JSON/HTML handoffs and model measurements performed by host tools](assets/workflow.png)

1. **Prepare evidence.** Read the existing activation archive, draw up to 40 candidates per amplitude interval, retain up to 256 scan points in proportion to the candidate counts, and use DeepSeek to label their observed contexts. This happens before the initial explanation.
2. **Interpret.** A fresh agent container loads `skills/interpret/` and reads only the target neuron's initial HTML and structured evidence.
3. **Experiment.** A fresh container loads `skills/experiment/`. Its tools run replacement sweeps, searches, native interventions, RelP attribution, family measurements, and blind maximum/minimum tests on the host. Saved input prefixes and scoring positions are retained.
4. **Revise.** A fresh container loads `skills/revise/`, reviews the measurements, and revises the explanation, Criticism, and captions. Fixed code assembles the report and cost records.
5. **Publish.** Export the generated report. If a local reference snapshot is available, publication adds a collapsible side-by-side comparison with aligned sections and linked scrolling. Reference content is added only at publication; it is not given to the research agents.

The outer framework is an executable shell, not another skill. Each agent reads one mounted skill; tools handle paths, GPU allocation, accounting, and HTML generation. The default investigator remains `deepseek-flash` with `reasoning_effort: high`.

**Semantic Clusters are separate from raw NMF routes.** The experiment agent explicitly assigns measured graph members to semantic clusters, which may merge or split routes. It must retain weak, contrary, mixed, and unresolved cases. Fixed code derives counts, attribution-mass shares, upstream summaries, and examples. Raw route membership is preserved, and revision does not replace clusters merely by renaming routes.

## 🖼️ Report Excerpt <a id="reports"></a>

<p align="center">
  <img src="assets/report-anatomy.png" width="560" alt="Raw gate versus up for selected Qwen3-32B L1/N20535 positions, exported from an existing generated report">
</p>

*Inside the neuron, Qwen3-32B L1/N20535.* Selected corpus and experimental token positions, plotted by raw gate and up. Orange, blue and grey denote positive, negative and mixed or unresolved semantic groups; they do not set an individual point's activation sign. The shaded “closed” half can still have nonzero `SiLU(gate) × up`. This plot is exported unchanged from a saved report; these samples do not estimate corpus density.

A full report also includes activation examples, token statistics, replacement experiments, attribution graphs, interventions, Criticism and an independent review. Fixed code plots the measurements; agents write the explanations and captions. The [report guide](docs/reports.md) explains how to read these sections and where their data comes from.

## 🚀 Quick Start <a id="quick-start"></a>

### 1. Try the report renderer locally

The demo generates a small synthetic fixture; it is not an experimental result. It needs no API key, Docker, GPU, or downloaded model and also works on macOS.

```bash
git clone https://github.com/RUC-TrustworthyAI/Neuron-Investigator.git
cd Neuron-Investigator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt
python examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

Open **http://localhost:8000/** to view the demo.

For CPU development checks, see [docs/development.md](docs/development.md). Tests do not add acceptance gates to the research workflow.

### 2. Configure a real investigation

- Linux, Bash 4+, Docker, and Python 3.10+ for the full pipeline.
- A CUDA-capable host with local dense-Qwen3 model weights. The default GPU group is `[0,1,2,3]`; a group holds one model sharded across its GPUs. The research setup used four A100 40 GB GPUs, but memory use depends on model and prefix length.
- An existing activation archive and per-neuron initial report packages; see [the input contract](docs/data.md). This repository does not collect a new corpus or download weights automatically.
- A DeepSeek API credential on the host for agents and auxiliary labeling/writing calls.

The agent container does not need CUDA. Model computation runs through host tools, while container inputs are mounted read-only.

```bash
python -m pip install -r requirements/gpu.txt
# Match the CUDA wheel to your NVIDIA driver. This is the research environment:
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
docker build -t neuron-investigator:local .
cp settings.json settings.local.json
```

Edit `settings.local.json` to point to your inputs:

```json
{
  "paths": {
    "reports": "/path/to/initial-reports",
    "activations": "/path/to/activation-archive",
    "model": "/path/to/Qwen3-32B",
    "state": "/tmp/neuron-investigator",
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

`prepare` freezes source and settings without calling an LLM or running the model. `execute` performs evidence preparation and the three stages using that snapshot. `publish` adds presentation controls while keeping the generated scientific content unchanged. Exemplars and cluster examples display up to 128 saved tokens before the first highlighted activation; the full measured prefix remains in the data.

The budget applies to all API calls in that batch and stays in host accounting. The default is USD 2; set `NEURON_PAPER_BUDGET_USD=unlimited` explicitly for no cap. Prices in settings are estimates inherited from the experiment environment; configure them for your endpoint. They are not account invoices.

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

These directories stay local and are ignored by Git. Existing results are preserved. Fresh interpretations clear old explanations and official labels. Scientific text is produced by the agents; the pipeline does not manually rewrite it during publication.

## 📚 Documentation <a id="documentation"></a>

| Guide | Start here when you want to… |
|---|---|
| [Running](docs/running.md) | Install dependencies, configure GPUs and credentials, run or continue a batch |
| [Input data](docs/data.md) | Prepare an activation archive and target report package |
| [Method and implementation](docs/method.md) | Trace the workflow to code and understand measurement scope |
| [Report guide](docs/reports.md) | Find the source and generation stage of a particular plot or section |
| [Development](docs/development.md) | Run CPU software checks and the offline demo |
| [Release notes](docs/release.md) | Review the packaged implementation and changes |

<details>
<summary><strong>Repository layout</strong></summary>

```text
run.sh                  Container orchestration and serial handoffs
settings.json           Scientific settings and configurable host paths
Dockerfile              Agent image with pinned harness, jq and ripgrep
skills/                 interpret, experiment, revise
tools/                  Evidence, GPU measurements, rendering and accounting
analysis/               Initial report builders and data-free HTML/JS templates
assets/                 Project banner, workflow diagram and report excerpt
requirements/           Host and development dependencies
tests/                  Software tests and synthetic fixtures
examples/demo.py        Synthetic offline renderer example
docs/                   Setup, data contracts, methods and report guide
```

The source includes the latest semantic-cluster interface, resumable scan annotations, shared context trimming, and publication-time comparison. Packaging changes make paths, credentials, dependencies, and image names configurable; they do not change the scientific sampling, scoring, attribution, or intervention rules. See [release notes](docs/release.md).

</details>

## 🤝 Contributing <a id="contributing"></a>

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [development guide](docs/development.md). Keep changes to measurement methods distinct from presentation and packaging changes. Update the pipeline and regenerate reports from saved inputs; do not edit scientific outputs by hand.

## 🙏 Acknowledgements <a id="acknowledgements"></a>

The report structure follows [Goodfire's neuron reports](https://already-clean.goodfire.com/). RelP adapts [TransluceAI/circuits](https://github.com/TransluceAI/circuits), with the source revision recorded in `tools/relp-source.json`. Agents use the official [DeepSeek Harness SDK](https://github.com/deepseek-ai/deepseek-harness/tree/master/python/sdk). The skill folder convention was informed by [AREX-Skill](https://github.com/VectorSpaceLab/AREX-Skill/tree/main/examples). Repository presentation also follows [Search-SWE](https://github.com/VectorSpaceLab/Search-SWE).

## 📄 License & Provenance <a id="license"></a>

Original project code uses [MIT](LICENSE). Third-party components and inherited renderer provenance are documented separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
