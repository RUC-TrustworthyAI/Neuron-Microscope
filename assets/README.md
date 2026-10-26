# Documentation figures

`hero.svg` / `hero.png` contain the project banner. `badges/*.svg` are the README navigation badges.

`workflow.svg` is the editable data-flow diagram; `workflow.png` is its README export. It shows evidence preparation, the three container stages, and host model execution.

`report-anatomy.svg` / `report-anatomy.png` show the **Inside the neuron** plot for Qwen3-32B L1/N20535. The plot illustrates the report format using selected measured positions.

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
