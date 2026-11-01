<p align="center">
  <img src="assets/neuron-microscope.png" alt="Neuron Microscope" width="100%">
</p>

<h1 align="center">Neuron Microscope</h1>

<p align="center">
  <strong>Investigating native neurons in language and vision models.</strong>
</p>

<p align="center">
  An evidence-first framework for reproducing and stress-testing neuron explanations.
</p>

<p align="center">
  <a href="README_zh.md">中文</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/">Language model reports</a> ·
  <a href="vision/README.md">Vision model visualizations</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1">Open the Colorado case</a> ·
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en">Watch the investigation</a> ·
  <a href="#use-neuron-microscope">Run the project</a>
</p>

<p align="center">
  <a href="docs/method.md"><img src="assets/badges/model.svg" alt="Model: dense Qwen3"></a>
  <a href="docs/workflow.md"><img src="assets/badges/pipeline.svg" alt="Pipeline: structured"></a>
  <a href="docs/reports.md"><img src="assets/badges/reports.svg" alt="Reports: interactive HTML"></a>
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en"><img src="assets/badges/live-demo.svg" alt="Showcase: online"></a>
  <a href="#use-neuron-microscope"><img src="assets/badges/quickstart.svg" alt="Usage: install and run"></a>
</p>

<a id="overview"></a>

Neuron Microscope reproduces and tests the neuron analyses behind Nick Cammarata's [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/). The project includes **Qwen3-32B language-model investigations** and **InceptionV1 feature visualization**.

| Model | What you can inspect | Entry point |
| --- | --- | --- |
| **Qwen3-32B** | Activation evidence, explanations, controlled tests, interventions and unresolved cases. | [Interactive neuron reports](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/) · [Investigation code](language/) |
| **InceptionV1 / Lucent Inception5h** | Five visualization objectives, natural-image examples, heatmaps and reference comparisons. | [Vision guide and examples](vision/README.md) · [Optimization code](vision/vision-repro/) |

**Data:** The current 50 reports use **Reasoning Mix 100M**. The [data guide](docs/data.md) describes the corpus sources, saved activation format and readers. Reports include selected contexts and measurements; full sampled texts and activation archives do not yet have a public download.

> [!IMPORTANT]
> **This is an independent reproduction effort.** In LLM investigations, the research models do not see AlreadyClean's explanations. The reference report is added only when the final report is published for comparison.

## Contents

- [Language model neuron reports](#language-model-neuron-reports)
- [Vision model neuron visualizations](#vision-model-neuron-visualizations)
- [Which AlreadyClean claims have we tested?](#which-alreadyclean-claims-have-we-tested)
- [The LLM investigation pipeline](#the-investigation-pipeline)
- [Evaluating the report evidence](#evaluating-the-report-evidence)
- [Observations from the current comparisons](#observations-from-the-current-comparisons)
- [Installation and usage](#installation-and-usage)
- [Recorded runtime and cost](#recorded-runtime-and-cost)
- [Limitations and future work](#limitations-and-future-work)
- [How to contribute](#how-to-contribute)
- [Related research from our group](#related-research-from-our-group)
- [Setup, data and report guides](#setup-data-and-report-guides)

## Language model neuron reports <a id="side-by-side-interactive-comparisons"></a> <a id="explore-the-evidence"></a> <a id="reports-and-walkthrough"></a><a id="live-demo"></a><a id="reproduction-results"></a>

### A side-by-side case: the Colorado neuron

Each neuron link opens the original report on the left and our reproduction on the right. Scroll either pane to move both; close the comparison to read the reproduction alone.

[Open the interactive comparison →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1">
    <img src="assets/colorado-comparison.jpg" alt="Colorado reference and Reasoning Mix 100M reproduction shown side by side" width="92%">
  </a>
</p>

Both reports identify a strong Colorado-related positive response, but they make different claims about what the neuron computes:

- **Published report:** the gate opens before a place-specific fill-in value, while the up projection carries signed place identity.
- **Our reproduction:** positive responses follow Colorado place and institution names across local-government, education, health, contact and news contexts. The negative account is limited to certain non-Colorado administrative contact and citation blocks.

In the Reasoning Mix 100M investigation, contact variants in `Contact block: {slot} End of contact.` reach **+188 to +352** for six Colorado institutions. Each value is the signed largest absolute response within the replacement span; all six peak on a space. Three out-of-state variants range from **-26.5 to +42.75**, while a generic clinic with a Colorado `720` number still reaches **+165**. These results support a Colorado-related response and leave the contributions of institution names, area codes and formatting to be separated. The report includes the full texts, attribution results, interventions and counterexamples.

### Browse all current reports

Explore **50 Qwen3-32B neurons on Reasoning Mix 100M**, with Blind and Search scores alongside each report.

[Open the report directory →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/">
    <img src="assets/report-directory.png" alt="Directory of Qwen3-32B neuron comparisons" width="92%">
  </a>
</p>

Each row links to a full interactive report with activation evidence, exemplars, searches, clusters, effects, tests, criticism, an independent review and cost records. Across the 49 available scores in each column, **Blind has mean 0.864 and median 0.891; Search has mean 0.782 and median 0.808**. The directory shows these summaries in the column headings.

## Vision model neuron visualizations <a id="vision-neuron-visualizations"></a>

The [vision module](vision/README.md) optimizes images for individual InceptionV1 channels and retrieves natural examples from an ImageNet scan. Its catalog contains **12 layers and 5,804 channels**. It implements Positive, Path-guided +, Diversity, Negative and Path-guided − views; the catalog size describes available targets, not the number of completed results.

[Browse vision comparisons →](vision/docs/gallery.md) · [Install and run →](vision/README.md#installation-and-usage)

<table align="center">
  <tr><th>Positive</th><th>Diversity</th><th>Negative</th></tr>
  <tr>
    <td><a href="vision/docs/assets/mixed4c_260/positive.png"><img src="vision/docs/assets/mixed4c_260/positive.png" width="200" alt="InceptionV1 mixed4c:260, optimized positive response"></a></td>
    <td><a href="vision/docs/assets/mixed4c_260/diversity.png"><img src="vision/docs/assets/mixed4c_260/diversity.png" width="200" alt="InceptionV1 mixed4c:260, four diversity members"></a></td>
    <td><a href="vision/docs/assets/mixed4c_260/negative.png"><img src="vision/docs/assets/mixed4c_260/negative.png" width="200" alt="InceptionV1 mixed4c:260, optimized signed negative response"></a></td>
  </tr>
</table>

These are saved outputs for **mixed4c:260**, using configuration A, 224 × 224 inputs, 1,024 steps and seed 0. Positive maximizes post-ReLU activation; Negative minimizes signed pre-ReLU activation. [Recorded settings and scores](vision/docs/assets/provenance.json) accompany the images.

The implementation pins a **Lucent Inception5h checkpoint**. The [reference metadata](https://already-clean.goodfire.com/microscope/inceptionv1/meta.json) also identifies Lucent pretrained weights; exact weight identity has not been verified. The [vision method guide](vision/docs/methodology.md) explains the objectives and sampling differences. The LLM workflow below applies to Qwen3.

## Which AlreadyClean claims have we tested?

The original essay contains claims at several levels. The table below tracks our language-model investigations and separates what we have directly examined from what remains open.

| Claim or question | Operational evidence needed | Current status |
| --- | --- | --- |
| Can native Qwen3 neurons receive coherent explanations? | Activation coverage, independent written probes, controlled tests and interventions. | **Preliminary evidence:** the 50 Reasoning Mix 100M reports include clear, partial and divergent cases. |
| Do the original neuron interpretations reproduce independently? | Blind reproduction followed by side-by-side comparison. | **Case-dependent.** The 50-report directory exposes both explanations and their Blind and Search results. |
| Does corpus choice change the visible behavior? | Matched sampling across corpora. | The current 50-report set uses **Reasoning Mix 100M**; this set alone does not isolate a corpus effect. |
| Are weak activations globally unimportant? | Thresholded, whole-network ablations across tasks and layers. | **Planned**, with no clear conclusion yet. |
| Does recursive generalization explain apparently mixed behavior? | Evidence that branches share a developmental or causal mechanism. | **Open.** Current reports can generate candidate cases but do not establish the general claim. |
| Are native neurons more useful than SAE features? | Matched neurons/features, data, tests, intervention budgets, and evaluation criteria. | **Planned.** No general comparison is claimed. |

> [!IMPORTANT]
> **We intend this table to become a living claims tracker. New evidence should link to a concrete report, model, corpus, intervention, or counterexample.**

## The LLM investigation pipeline <a id="the-investigation-pipeline"></a> <a id="the-three-stage-investigation-pipeline"></a> <a id="how-an-investigation-works"></a> <a id="workflow"></a>

### Follow a recorded investigation

[Open the recorded language model walkthrough →](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en)

<p align="center">
  <a href="https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en#loops">
    <img src="assets/pipeline-walkthrough-en.jpg" alt="Language model walkthrough showing three explanation and blind-labeling rounds" width="92%">
  </a>
</p>

The walkthrough follows **L55_N11703 on Reasoning Mix 100M** from saved evidence to the final report. It presents the task prompts, excerpts of their inputs, structured responses, all three explanation-and-labeling versions, measured experiments and costs. The selected explanation is frozen before text construction and GPU experiments.

### Current runnable workflow

The code in `language/` orchestrates direct LLM API calls and target-model measurements. The LLM returns explanations, predictions and experimental inputs in defined JSON formats; Python selects evidence, executes measurements and assembles HTML. The configuration generator supplies the settings below.

1. **Sample evidence.** Allocate 500 positions equally across nonempty activation bins, selecting within each bin from the full saved corpus. Redistribute a bin's unused quota only when its corpus population is too small. The archive retains gate, up and product channels; the explanation model receives product values, contexts, extreme tokens and document exemplars, quantile cases, token selectivity, peak statistics and direct logit effects.
2. **Explain, check and freeze.** Request at most five response rules per pole. The explanation and rule descriptions express the same account. A labeler predicts the sampled positions with their measurements hidden, in five concurrent batches of up to 100. Two revisions each read the preceding explanation, the 500 measured contexts and focused error feedback, followed by another blind labeling pass. Among the three versions, select the highest coverage with directional accuracy **>90%**; if none qualifies, select the highest accuracy. Freeze that explanation together with its own labels. An unsupported pole may remain unexplained.
3. **Measure.** Run three feedback-guided Search rounds, requesting six regions with 16 texts each. Design four controlled tests with ten conditions each, measure them and improve the test design once. The head figure comes from these controlled tests. A separate Blind writer receives only the frozen explanation and rules, with six texts per supported pole. Search and Blind score full-input maxima or minima; controlled tests score a replacement span, a common suffix or a specified token. Batched GPU operations also compute attribution, family responses and interventions. With the shared worker enabled, one loaded model serves queued operations from multiple runs while their API calls overlap.
4. **Analyze and publish.** The LLM analyzes support, counterexamples and unresolved patterns while preserving the frozen explanation and rules. An independent reader assesses the account. Fixed code renders the interactive report and, when a reference snapshot is supplied, adds the side-by-side comparison after research is complete.

**Activation spectrum and Inside the neuron share one sample pool and the predictions from the refined, frozen explanation.** The labeler reads the full observed prefix and nearby continuation as linguistic context. It assigns No visible cue when the explanation does not clearly predict either direction. Because these samples inform refinement, their agreement scores provide development feedback. Independent written probes and controlled experiments supply further tests.

See [the workflow](docs/workflow.md), [measurement details](docs/method.md), and [run configuration](docs/running.md).

## Evaluating the report evidence <a id="how-we-evaluate-an-explanation"></a>

The current directory reports **Blind score** and **Search score**. For each, average the available ratios of reproduced maximum / reference maximum and |reproduced minimum / reference minimum|. A score can exceed 1; missing measurements are excluded from its summary statistics. Blind tests use texts written from the frozen explanation alone, while Search uses measured feedback. These extrema measure response strength relative to the reference; causal claims require the experiments below.

| Dimension | Question | Evidence in the reports |
| --- | --- | --- |
| Coverage | Does the explanation account for both poles and the important residual cases? | Activation spectrum, exemplars, semantic clusters, prediction errors, and “No visible cue” cases. |
| Specificity | Does it identify the trigger rather than merely the surrounding topic? | Controlled replacements, counterexamples, searches, and blind tests. |
| Predictive value | Can the explanation anticipate responses on inputs not used to formulate it? | Independent explanation-only maximum/minimum tests, supported by controlled templates and family measurements. |
| Causal relevance | Does changing the neuron or its inputs have the predicted effect? | Native interventions, ablations, controlled input changes, and output measurements. |
| Stability | Does the interpretation persist across samples, prompts and independent runs? | Comparisons of explanations and measurements under matched conditions. |
| Uncertainty | What remains unexplained or underdetermined? | Explicit criticism, alternative accounts, weak poles, and failed tests. |

RelP provides supporting attribution evidence using modified derivatives; its graphs alone do not establish a native causal effect. See [the measurement scope](docs/method.md#relp-and-intervention-scope).

## Observations from the current comparisons <a id="what-we-have-found-so-far"></a> <a id="key-findings"></a>

### Shared concepts can conceal different response conditions

The following examples come from the current **Reasoning Mix 100M** set. Reading the explanations alongside their experiments reveals both shared concepts and differences in scope.

| Neuron | Shared interpretation | What still differs |
| --- | --- | --- |
| [L50_N3507](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L50_N3507/?compare=1) | Mathematical labels and variable introductions can be positive; rational-fraction expressions can be negative. | Our account limits the negative pole to rational/fraction forms; the reference generalizes to other form placeholders and emphasizes geometric object names. |
| [L24_N17907](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L24_N17907/?compare=1) | “Has/must be seen or experienced to be believed” expresses a required firsthand experience. | Our negative account includes subword and heterogeneous residual effects; the reference treats the shallow negative pole as a token offset. |
| [L57_N2065](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L57_N2065/?compare=1) | Arithmetic-related prose can be positive, while some institute affiliations are negative. | Our account also includes proof and theorem contexts; the reference emphasizes accumulated arithmetic context and separates several negative pockets. |
| [L45_N16800](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L45_N16800/?compare=1) | Explicit wordplay can produce positive responses. | Our account centers on pun asides; the reference requires a word's double reading to become visible and also covers transparent name origins. |

L45_N16800 illustrates why the shared topic alone is insufficient. Its saved corpus maximum is **95**, above the reference corpus maximum of **71.5**, yet its Blind maximum is only **9.125**, compared with **50.75** in the reference. This gap motivates contrasts between a pun flag and an actual double reading in context.

### Corpus extrema provide a target for written tests

For Colorado, the Reasoning Mix 100M corpus spans **-44.5 to +308**. Search reaches **-44.25 to +296**, the selected head figure **-26.5 to +352**, and Blind **-47 to +278**. The head figure scores the replacement span, while Search and Blind use full-input extrema. These measurements answer different questions and should be read with their texts and scoring rules.

Corpus extrema describe the saved scan and can be exceeded by constructed texts. A high written response shows that the test reached a strong activation; the controlled contrasts and interventions are needed to assess the proposed explanation.

The current reports leave uncertainty at three points: whether the corpus contains the relevant contexts, whether the explanation captures their response conditions, and whether the writer constructs strong tests from that explanation. Matched comparisons are needed to separate these effects.

## Installation and usage <a id="use-neuron-microscope"></a> <a id="quick-start"></a>

Choose an entry point for the model and evidence you want to inspect.

| Mode | What you can do | Requirements |
| --- | --- | --- |
| **Explore** | Inspect LLM reports, a recorded investigation and vision comparisons. | Browser only. |
| **Try the renderer** | Generate and view a small synthetic report fixture. This tests the software and interface, not a scientific result. | Python; no API key, Docker, GPU, or model download. |
| **Run an LLM investigation** | Analyze neurons from an existing activation archive and publish new reports. | Linux, Python 3.10+, CUDA host, model weights, activation data, and an LLM endpoint. |
| **Visualize a vision neuron** | Optimize images and export an InceptionV1 viewer. | Python 3.11+, a separate environment, model assets and a CUDA host for optimization; see [vision setup](vision/README.md#installation-and-usage). |

### 1. Try the offline renderer

```bash
git clone --branch hr-language-refactor https://github.com/RUC-TrustworthyAI/Neuron-Microscope.git
cd Neuron-Microscope

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r language/requirements/cpu.txt

python language/examples/demo.py --output runs/demo
python -m http.server 8000 --directory runs/demo
```

Open <http://localhost:8000/>. The generated content is synthetic and should not be treated as experimental evidence.

### 2. Configure a full investigation

The pipeline requires Linux, Python 3.10+, a CUDA host with local dense-Qwen3 weights, compatible saved activations, an activation-only report package, and an LLM endpoint. Install the measurement dependencies:

```bash
# Choose a PyTorch CUDA wheel compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r language/requirements/gpu.txt
```

Create `workflow.local.json` with the configuration generator. Supply the neuron, local model, activation-only report package, activation archive and GPU indices:

```bash
python language/examples/configure.py \
  --neuron L55_N11703 \
  --model /path/to/model \
  --report /path/to/activation-only/report \
  --activations /path/to/activation/archive \
  --gpus 0 1 2 3 \
  --output workflow.local.json
```

Review `llm.model`, reasoning effort, endpoint URLs and token prices in the generated configuration. The supplied settings use `deepseek-flash` with `high` reasoning; the recorded reports identify the model as `deepseek-v4.1-flash-1017`. Supply `DEEPSEEK_API_KEY` in the environment, or point `NEURON_PAPER_API_KEY_FILE` to a local credential file.

```bash
NEURON_MICROSCOPE_GPU_WORKER=1 python -m language \
  --config workflow.local.json --output runs/example
python -m http.server 8000 --directory runs/example/report
```

Append `--stop-after interpretation` to inspect the refined explanation and both labeled plots using saved activations before loading the target model. Repeating the command with the same configuration reuses completed API responses and saved sampling and measurement records. Use a new output directory when changing scientific inputs.

For concurrent neurons, use separate configurations and output directories, with the same `gpu_lock` for runs sharing a GPU group. `NEURON_MICROSCOPE_GPU_WORKER=1` enables the shared resident model. The supplied `llm.max_concurrent_requests: null` places no application-level API concurrency cap. The [run guide](docs/running.md) covers continuation, comparison export and execution records.

## Recorded runtime and cost <a id="reproducibility-and-cost"></a>

Each LLM investigation retains its configuration, API requests and responses, token-price estimates, sampled evidence, explanation and prediction versions, measurements, and final report. Fresh interpretations use the target activation evidence without previous explanations or official labels.

For the Colorado / **Reasoning Mix 100M** investigation shown in the language model walkthrough:

| Item | Recorded value |
| --- | --- |
| Target model | Qwen3-32B |
| Saved evidence | Reasoning Mix 100M; 500 sampled positions; 498 labeled positions in the selected version |
| Research and auxiliary calls | `deepseek-v4.1-flash-1017`, high reasoning effort |
| GPU group | 4 × NVIDIA A100 40 GB |
| Measured execution time | 25 min 59 s |
| GPU experiment time | 5 min 15 s of recorded GPU work; 0.351 GPU card-hours |
| API streaming time | 19 min 25 s |
| Estimated API cost | $0.6181 across 26 completed requests |
| Recorded elapsed time | 25 min 59 s |

GPU time includes recorded model loading and operations. API streaming time counts overlapping streams once. GPU and API work can overlap, so their durations do not sum to execution time.

These values describe one measured case. Costs exclude activation collection and depend on the configured model endpoint and token prices. The underlying breakdown is available in the [published report](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=0#interpretation-costs) and [walkthrough](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en#report).

## Limitations and future work

### Current limitations

- The current set contains 50 neurons with reference reports, so its results cannot be generalized to all Qwen3-32B neurons.
- Full activation archives do not yet have a public download; external users currently need to supply compatible input data.
- The offline demo is synthetic and does not reproduce a published scientific result.
- Corpus coverage, prompt choice, evidence presentation and LLM stochasticity can change an interpretation.
- Blind and Search scores compare activation extrema. They do not establish mechanistic correctness, and a mean over available poles may summarize only one pole.
- Negative poles and weak activations are often less clearly explained than strong positive responses.
- We have not completed matched native-neuron versus SAE comparisons.

### Future work <a id="future-work"></a>

- [ ] Publish a compact, real-neuron evidence pack that can regenerate one scientific report without the full model or activation archive.
- [ ] Extend beyond neurons with published reference reports to a broader random sample of the model.
- [ ] Systematically measure explanation stability across independent runs, prompts, samples and corpora.
- [ ] Add matched native-neuron and SAE-feature comparisons under the same evaluation protocol.
- [ ] Test the effect of weak activations on global behavior through thresholded, whole-network ablations across tasks and layers.
- [ ] Test major AlreadyClean claims such as recursive generalization.
- [ ] Support additional models, layers, languages, and genres.

## How to contribute <a id="contribute"></a>

We welcome evidence that strengthens, narrows, or contradicts an explanation.

Useful contributions include:

- **A new neuron case:** model, layer/neuron ID, corpus, sampling procedure, and a complete report.
- **A counterexample:** the tested context, scoring position, measured response, and the explanation it challenges.
- **A controlled experiment:** matched inputs that distinguish two plausible accounts.
- **A new model or corpus:** compatible activation data with provenance and redistribution terms.
- **An evaluation improvement:** a more reliable test of coverage, prediction, causal relevance, stability, or uncertainty.
- **A pipeline improvement:** clearer evidence presentation, more robust tools, lower compute requirements, or better reproducibility.

Please open a [GitHub Issue](https://github.com/RUC-TrustworthyAI/Neuron-Microscope/issues) with a concrete neuron ID, context, experiment, result, or proposal. Reproducible examples are especially valuable, including failures and disagreements.

## Related research from our group <a id="related-work"></a>

Our group's related work spans feature learning, concept-based analysis, explanation evaluation, and LLM safety. Publication details are also listed on [Xiting Wang's faculty page](https://ai.ruc.edu.cn/szdw/zrjs/wxt/index.htm).

| Work | Connection | Links |
| --- | --- | --- |
| **C²R: Cross-sample Consistency Regularization Mitigates Feature Splitting and Absorption in Sparse Autoencoders** — Haoran Jin et al., ICML 2026 | Encourages consistent latent assignments across samples to reduce feature splitting and absorption in SAEs. | [Paper](https://arxiv.org/abs/2606.30609) · [Code](https://github.com/hr-jin/Cross-sample-Consistency-Regularization) |
| **Uncovering Safety Risks of Large Language Models through Concept Activation Vector** — Zhihao Xu et al., NeurIPS 2024 | Uses safety concept activation vectors to investigate model safety mechanisms and vulnerabilities. | [Paper](https://arxiv.org/abs/2404.12038) · [Code](https://github.com/SproutNan/AI-Safety_SCAV) |
| **Evaluating Readability and Faithfulness of Concept-based Explanations** — Meng Li et al., EMNLP 2024 | Formalizes concept-based explanations and develops measures of readability and faithfulness. | [Paper](https://aclanthology.org/2024.emnlp-main.36/) · [Code](https://github.com/hr-jin/Concept-Explanation-Evaluation) |
| **Do LLMs Know Their Vulnerable Scenarios?** — Ziheng Peng et al., arXiv 2026 | Uses SAE concepts and causal attribution to identify scenarios that suppress refusal. | [Paper](https://arxiv.org/abs/2607.23496) |

## Setup, data and report guides <a id="documentation"></a>

| Guide | Start here when you want to… |
| --- | --- |
| [LLM installation and running](docs/running.md) | Install dependencies, configure GPUs and credentials, and run or continue a batch. |
| [LLM input data](docs/data.md) | Prepare an activation archive and target report package. |
| [LLM method and implementation](docs/method.md) | Trace the workflow to code and understand the measurement scope. |
| [LLM report guide](docs/reports.md) | Find the source and generation stage of a particular plot or section. |
| [LLM development](docs/development.md) | Run CPU software checks and the offline demo. |
| [LLM prompts and output contracts](language/workflow/) | Inspect the explanation, labeling, experimental-design and report-analysis requests. |
| [Vision setup and examples](vision/README.md) | Install the InceptionV1 module, inspect visualizations and export a viewer. |

## Citation

If you use Neuron Microscope in academic work, please cite the repository and the specific release or report used:

```bibtex
@software{ruc_trustworthyai_neuron_microscope_2026,
  author = {Haoran Jin and Ziheng Peng and Xiting Wang},
  title = {Neuron Microscope: Reproducing and Stress-Testing Native Neuron Explanations},
  year = {2026},
  url = {https://github.com/RUC-TrustworthyAI/Neuron-Microscope}
}
```

## Acknowledgements

The report structure follows [Goodfire's neuron reports](https://already-clean.goodfire.com/). RelP adapts [`TransluceAI/circuits`](https://github.com/TransluceAI/circuits), with the source revision recorded in [`language/tools/relp-source.json`](language/tools/relp-source.json).

## License

The LLM implementation is released under the [MIT License](LICENSE). The imported vision module retains its [source and licensing notice](vision/THIRD_PARTY_NOTICES.md); its original code has no declared upstream license. Third-party components retain their original licenses.
