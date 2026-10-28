# Live Demo

**English** · [简体中文](live-demo_zh.md)

**[Neuron reports](https://neuron-microscope-papers.tangysiren10.chatgpt.site/)** · **[Pipeline walkthrough](https://neuron-microscope-pipeline.tangysiren10.chatgpt.site/?lang=en)**

Both websites are publicly accessible.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open **http://127.0.0.1:8000/?lang=en** in a current Chrome, Edge, Firefox or Safari browser. English is the default. Use the language switch to select Chinese, or open **http://127.0.0.1:8000/?lang=zh**. The current step is retained when switching languages. The demo uses static HTML, CSS and JavaScript. It needs no API key, model, GPU, Docker or Python packages.

![Neuron Microscope recorded walkthrough](../assets/live-demo-en-hd.png)

## What the demo shows

Follow a recorded investigation of Qwen3-32B neuron **L55/N11703**, from its initial evidence page through interpretation, experiments, revision and the final interactive report. Each step contains an explanation, saved task and skill instructions, selected prompt examples, key operations, and corresponding artifacts. Screenshots can be enlarged; saved reports retain their plots, hover details and folding controls.

The walkthrough interface and explanations are bilingual. Recorded prompts, sample contexts, measurements and scientific reports retain their source content.

The recording includes an interrupted experiment stage, reuse of an earlier completed measurement stage, and a later labeling refresh. These events are shown in sequence. It reflects the pipeline version used for that recording.

The demo includes selected excerpts and report resources. Complete API request/response logs, credentials, model weights and activation archives stay outside this website. Scientific prose, displayed measurements and sample contexts are preserved by the exporter. The report resources omit local provenance paths and a duplicate example collection, and are compressed for static delivery. They load in the browser with `DecompressionStream`.

## Optional GitHub Pages hosting

For a repository with GitHub Pages available:

1. Open **Settings → Pages** and select **GitHub Actions** as the source.
2. Open **Actions → Publish live demo → Run workflow**, choose `main`, and run it.
3. Open the URL shown in the `github-pages` deployment. Add `?lang=en` for English or `?lang=zh` for Chinese.

The workflow is manual. Committing demo changes does not trigger a deployment.

## Refresh the recorded case

To export another prepared local recording:

```bash
python3 examples/export_live_demo.py --source /path/to/local-recording
```

This regenerates the curated files under `website/content/` and compressed report resources under `website/reports/`. The website interface and screenshot selection are maintained separately. The command reads saved material and performs no API calls or model runs.
