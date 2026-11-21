# Comparing matched conditions in Anatomy

The paired view shows how changing one part of a sentence affects a neuron's response. It puts two measured conditions from the same replacement sweep side by side and shows the raw gate, up projection and native activation at every token. The saved activation is `SiLU(raw gate) × up` before the down projection.

## Colorado reference

Goodfire's Qwen3-32B Colorado report saves ten variants of a civic-office phone template and shows Denver and Colorado side by side. At the shared opening parenthesis before the phone number, the reported gate, up and activation are `14 / 12.75 / 178.5` for Denver and `17.25 / 13.5 / 232.875` for Colorado. Each original strip scales a channel within its own sentence. This implementation uses the same scale for each channel across both sentences so that their heights can be compared directly. [Colorado Anatomy](https://already-clean.goodfire.com/neuron-scientist/L55_N11703/#anatomy)

## Selecting and reading a pair

1. Use two completed conditions from one retained sweep. Match the saved measurement IDs and retain the shared template, each replacement, the exact tokenization and the scoring rule.
2. Prefer a pair with the same scored token at the same location in the unchanged suffix. The next preference is the same token at a corresponding position within the replacement. If only separately scored positions are available, identify both positions explicitly. Within each category, prefer the headline sweep and then the largest absolute difference in recorded activation. A pair is eligible only when its tokenized inputs differ at or before both scored positions.
3. Read the three signed traces and the highlighted scoring positions. Replacement length can shift token indices between conditions. The anchor comparison records raw gate, up and native activation for each side. The caption describes which recorded signals changed at those positions. Channel-specific causal effects would require a separate intervention. [SwiGLU sign regimes](https://already-clean.goodfire.com/#:~:text=Separating%20the%20Sign%20Regimes)

## Render and verify

A complete [structured workflow](workflow.md) adds the pair automatically when it has eligible evidence. To add it to an existing investigation, reuse its completed measurements and report in a separate output directory. The command requires no model forward or explanation API call.

```bash
python -m language \
  --anatomy-from runs/l15-n4898-fresh-20261113/investigation \
  --output runs/task3-l15-n4898
```

Open `report/index.html#anatomy`. The pair is saved in `anatomy-pair.json` and in the report data at `report/paper-data.json`. `anatomy-source.json` records the source directory and neuron; `status.json` records the render outcome. Repeating the command rebuilds the view from the current saved source, without source-file hashes, and rejects an output directory owned by another investigation.

Check that both case IDs resolve to one saved sweep. Compare each token ID, offset, raw gate, up value and native product to the saved measurement. Verify the highlighted score position and confirm the caption uses the same token only when the saved positions correspond. For example, the L15_N4898 closing-parenthetical sweep provides different-length insertions followed by the same scored comma.
