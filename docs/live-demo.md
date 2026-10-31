# Recorded LLM investigation

**English** · [简体中文](live-demo_zh.md)

**[Neuron reports](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/)** · **[Recorded walkthrough](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/?lang=en)**

The walkthrough records **Qwen3-32B L55/N11703 on 29 October 2026**, using Diverse 100M and 500 sampled positions. It preserves that run's interpretation, labeling, experiments, post-experiment revision and report review. The [current runnable workflow](workflow.md) freezes the explanation before experiments and analyzes the results against it. The saved recording documents its own procedure and measurements.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open <http://127.0.0.1:8000/?lang=en> in a current Chrome, Edge, Firefox or Safari browser. English is the default; select Chinese with the language switch or `?lang=zh`. Switching languages retains the current step. The static preview needs only Python's HTTP server and a browser.

![Recorded Colorado investigation](../assets/pipeline-walkthrough-en.jpg)

## Read the evidence

The six steps cover evidence preparation, interpretation, sample labeling, experiments, revision, and review/publication. Each step links to saved tasks, skill instructions, selected prompts, key operations and corresponding artifacts. Screenshots enlarge on click; saved reports retain their plots, hover details and folding controls.

The interface and explanations are bilingual. Prompts, sample contexts, measurements and scientific reports retain their source content. The final report plots the 495 labeled positions from the shared 500-position sample.

`website/` contains the same recorded case as the linked online walkthrough. It includes curated excerpts and report resources. Complete API logs, credentials, weights and activation archives stay outside the website. Report resources are compressed and load through the browser's `DecompressionStream`.

## Host this walkthrough on GitHub Pages

For a repository with GitHub Pages available:

1. In **Settings → Pages**, choose **GitHub Actions** as the source.
2. Open **Actions → Publish live demo → Run workflow** and select the branch containing these `website/` files.
3. Open the deployment URL. Add `?lang=en` or `?lang=zh` to select a language.

The workflow publishes the walkthrough in `website/` and is triggered manually. The neuron-report directory and vision viewer have their own export and hosting entries.

## Import a prepared public recording

Use a prepared walkthrough containing `index.html`, its JavaScript and CSS, and the `content/`, `reports/` and `screenshots/` directories:

```bash
python3 examples/export_live_demo.py --source /path/to/prepared-walkthrough
```

The command synchronizes those display assets into `website/`, preserving saved report and measurement bytes. `--output` selects another destination. Preparing the public snapshot includes selecting prompt excerpts, screenshots and report resources; the import step performs no API calls or model measurements.
