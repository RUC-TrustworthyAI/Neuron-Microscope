# Documentation figures

`neuron-microscope.svg` / `neuron-microscope.png` contain the project banner. `badges/*.svg` are the README navigation badges.

`workflow.png` illustrates the three-agent research workflow using the Colorado neuron: interpreting evidence, running experiments on Qwen3-32B, and revising the report.

`report-anatomy.svg` / `report-anatomy.png` show the **Inside the neuron** plot for Qwen3-32B L1/N20535. The plot illustrates the report format using selected measured positions.

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
