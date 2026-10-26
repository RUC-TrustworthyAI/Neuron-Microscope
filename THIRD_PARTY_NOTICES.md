# Third-party provenance and distribution status

The root MIT license covers original Neuron Investigator code. It does not override third-party rights. This repository is currently a private release candidate; these unresolved items must be addressed before a public release that includes them.

| Material | Source | Status |
|---|---|---|
| Reference HTML/CSS/JavaScript in `analysis/report.html` | Goodfire neuron-scientist pages, originally the L55_N11703 page at https://already-clean.goodfire.com/neuron-scientist/L55_N11703/ | Adapted presentation template. No explicit redistribution license has been confirmed. Obtain permission or replace the inherited renderer before public redistribution. Official neuron answers are not included in the template. |
| RelP implementation in `tools/relp.py` | TransluceAI/circuits, commit `2d215d4ba016ba8602b69d71fc0f1ad139a427b7`; see `tools/relp-source.json` | Local HF adaptation of the cited backward rules and edge algorithm. No repository-root license was found in the pinned tree. The frontend asset license is not a license for tracing code. Confirm permission / applicable code license before public redistribution. |
| DeepSeek Harness SDK and runtime | https://github.com/deepseek-ai/deepseek-harness | Installed from public PyPI packages in Docker, not vendored. Respect the upstream MIT license and bundled dependency notices. |
| Skill directory convention | https://github.com/VectorSpaceLab/AREX-Skill/tree/main/examples | Structural reference for `SKILL.md`, `references/`, `scripts/`; task instructions are project-specific. |
| PyTorch, Transformers, tokenizers, NumPy, SciPy, PyArrow, Accelerate | Their respective published packages | External dependencies, not vendored. Their licenses apply separately. |
| Model checkpoints and corpus text | User-provided local inputs | Not distributed in this repository. Model/data licenses must be followed by users and in any future data release. |

The report source may include adaptations in the renderer patch modules as well as the base template; review those together when clearing or replacing the renderer. Public accessibility of an upstream page or repository is not treated as a redistribution grant.
