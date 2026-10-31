# Documentation figures

`neuron-microscope.svg` / `neuron-microscope.png` contain the project banner. `badges/*.svg` are the README navigation badges.

`workflow.png` is a schematic of the recorded Colorado investigation. Its numerical labels are illustrative. The linked reports and recorded walkthrough supply measured results together with their input templates and scoring positions. The current runnable procedure is described in [the LLM workflow guide](../docs/workflow.md).

`report-anatomy.svg` / `report-anatomy.png` show the **Inside the neuron** plot for Qwen3-32B L1/N20535. The plot illustrates the report format using selected measured positions.

`colorado-comparison.png` and `report-directory.png` are screenshots from the SlimPajama 100M reports. The comparison shows the original AlreadyClean report on the left and our independent report on the right. The directory image previews the comparison collection. Readers can open the [Colorado report](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/L55_N11703/) or [report directory](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/) to inspect the full pages.

`pipeline-walkthrough-en.jpg` and `pipeline-walkthrough-zh.jpg` show the experiment stage of the [recorded walkthrough](https://ruc-trustworthyai.github.io/Neuron-Microscope-Pages/walkthrough/?lang=en#agent2), captured on 30 October 2026 from the published site build. This recording follows a Colorado investigation using Diverse 100M.

`corpus-coverage.svg` plots the observed native product extrema, `SiLU(gate) × up`, for Qwen3-32B neuron L51_N20229. Values come from the saved corpus statistics used for the 29 October 2026 comparison runs, each with approximately 100M tokens: SlimPajama [-23.625, 56.25], FineWeb [-15.1875, 61.25], Common Pile [-19, 59], Diverse [-35, 98], and Reasoning Mix [-32.75, 111]. The endpoints describe those corpus scans, not theoretical activation bounds.

With Node.js and `sharp` installed, save the following as a `.cjs` script and run it from the repository root:

```javascript
const sharp = require('sharp');

async function main() {
  await sharp('assets/neuron-microscope.svg').png().toFile('assets/neuron-microscope.png');
  await sharp('assets/report-anatomy.svg', { density: 144 })
    .flatten({ background: '#fff' }).png().toFile('assets/report-anatomy.png');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
```
