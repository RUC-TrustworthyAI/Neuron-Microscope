# Documentation figures

`hero.svg` / `hero.png` are the project banner: an original branching N-shaped neuron mark on a light blue background. This is conceptual artwork, not a measurement. `badges/*.svg` are local navigation badges.

`workflow.svg` is the editable data-flow diagram; `workflow.png` is its README export. It shows evidence preparation, the three container stages, and host model execution. It contains no experiment results.

`report-anatomy.svg` is exported from the rendered **Inside the neuron** plot for Qwen3-32B L1/N20535 in the saved `anatomy-scan256-batch10-20261025T043403Z` report. `report-anatomy.png` is its raster export. Point coordinates, colours and labels are unchanged. The only export additions are the SVG namespace and an explicit font family; the original report is untouched. There are no corpus passages, hover payloads, activation arrays or API records in this asset. This individual plot illustrates the report format, not a population-level finding.

The banner and workflow diagram are original project artwork. The report export uses the inherited renderer described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

With Node.js and `sharp` installed, save the following as a `.cjs` script and run it from the repository root:

```javascript
const sharp = require('sharp');

async function main() {
  await sharp('assets/hero.svg').png().toFile('assets/hero.png');
  await sharp('assets/workflow.svg').png().toFile('assets/workflow.png');
  await sharp('assets/report-anatomy.svg', { density: 144 })
    .flatten({ background: '#fff' }).png().toFile('assets/report-anatomy.png');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
```
