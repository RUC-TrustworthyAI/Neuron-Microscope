# Documentation figures

`neuron-microscope.svg` / `neuron-microscope.png` contain the project banner. `badges/*.svg` are the README navigation badges.

The figures linked from the main READMEs use the current Reasoning Mix 100M reports:

- `colorado-comparison.svg` summarizes the published and reproduced Colorado explanations. Its contact-test values come from L55_N11703: six Colorado institutions reach +188 to +352; a generic clinic with a 720 area code reaches +165. This is a summary illustration.
- `report-directory.png` previews the 50-report directory with Blind and Search scores, their means and medians.
- `pipeline-walkthrough-en.svg` and `pipeline-walkthrough-zh.svg` outline the six steps in the language model walkthrough: evidence, initial explanation, blind labels and two revisions, selection and freezing, experiments, and report analysis.
- `colorado-activation-ranges.svg` plots L55_N11703 extrema: corpus [-44.5, 308], Search [-44.25, 296], head figure [-26.5, 352], and Blind [-47, 278]. The head figure scores the replacement span; Search and Blind score complete inputs.

The [Colorado report](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/L55_N11703/?compare=1), [report directory](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/reasoning-mix-100m-50/) and [language model walkthrough](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/language/?lang=en) provide the complete explanations, texts and measurements.

`report-anatomy.svg` / `report-anatomy.png` illustrate the Inside the neuron format using selected measured positions for Qwen3-32B L1/N20535.
