# Measuring a neuron's output effects

The Effect section asks how much one neuron's activation changes the next-token distribution at a particular position. Measure the target and same-layer controls on the same input, then compare effects across activation magnitudes and controlled conditions. The measurements characterize this neuron on the selected cases.

## Colorado reference

Goodfire's Qwen3-32B Colorado report measures 80 texts. It includes 20 high and 10 low corpus cases, 40 examples across 14 semantic groups, and 10 location variants of one controlled template. Its Effect table displays the target KL, the largest of three same-layer control KLs, the probability change of the next token in the text when available, and the largest token probability losses. Its summary gives the target median and the median pooled across all random-control measurements. [Colorado Effect](https://already-clean.goodfire.com/neuron-scientist/L55_N11703/#ablation)

The local design uses the same single-position intervention, random comparison and target size. It explicitly allocates cases to moderate and weak activation so their effects are visible in the report. The published report supplies examples and outcomes. Its exact selection rules and control seed are unavailable. Nick Cammarata discusses comparing native activation magnitudes relative to a neuron's own range in the accompanying essay. [Neural Networks Are Already Clean](https://already-clean.goodfire.com/#:~:text=Percentiles%20or%20raw%20activations)

## Sample selection

| Cohort | Target | Source and selection |
| --- | ---: | --- |
| Corpus high | 20 | Highest strictly positive saved token activations |
| Corpus low | 10 | Lowest strictly negative saved token activations |
| Moderate | 20 | Saved natural spectrum positions above 10% and at most 50% of the corresponding pole's corpus extreme |
| Weak | 20 | Saved natural spectrum positions at most 10% of the corresponding pole's corpus extreme, including zero |
| Controlled | 10 | Conditions from one retained replacement sweep with a common template and scoring rule |

Rank the first two cohorts by signed corpus activation. Retain their raw and relative magnitudes. A sparse pole's ranked tail can overlap the moderate magnitude range. For moderate and weak cases, use a reproducible draw that spreads coverage across available signs, semantic groups and source documents. Choose a controlled sweep that can supply the most distinct cases up to its quota, using the headline sweep to break ties. Retain the saved scoring position for each condition. When two cohorts share an input and scoring position, keep the ranked corpus case first, followed by controlled, moderate and weak cases. Record actual coverage when a cohort falls short.

If the saved corpus ranking or extrema are unavailable, use the saved natural spectrum pool as the corresponding source and record that scope in the plan. Cohort membership uses the archived activation. Each ablation also records the activation from its unmodified forward.

## Matched interventions

1. For each natural or controlled case, reconstruct the exact saved token IDs through the selected position. If a following token exists in the saved input, record it separately from the model input. The following token is an observed continuation for a corpus document or an authored continuation for a controlled condition.
2. On the same dense Qwen3 checkpoint, device group and precision, run an unmodified forward and then zero only the target neuron's native SwiGLU product at the selected position, before the MLP down projection. Use the output logits at that same position for the next-token distribution.
3. Draw three distinct random neuron indices from the target's layer using the configured seed and neuron coordinates. Exclude the target. Evaluate each random neuron independently using the same input and position, and keep the same sampled indices across cases.
4. Compute `KL(P_base || P_zero)` in natural-log units for the target and each control. Retain the top predictions, the largest signed probability changes, and the before/after probability of the following token from the saved input. The report identifies the continuation as corpus or authored text. When the selected position ends the saved input, compare the unmodified model's top prediction with the probability of that same token after intervention.

The paired target and random-control readouts measure the response of the output distribution. A large KL shows sensitivity to the edit. The direction of the change comes from the continuation and token-level probability changes. Compare the distributions within each sampled cohort. The overall median describes this selected mixture of cases.

## Run and verify

Use the Linux CUDA environment in the [run guide](running.md#environment). A complete [structured workflow](workflow.md) includes the ablation stage. To extend an existing investigation, use its completed run directory and a separate output directory.

```bash
python -m language \
  --ablate-from runs/qwen3-17b-real-reasoning-mix/investigation \
  --output runs/qwen3-17b-l6-n1169-ablation
```

The extension reuses the source investigation's model, settings, saved evidence and rendered report. It runs the new model interventions and updates Effect using the saved report. The command needs no explanation API credentials. Rerun the same command to reuse completed measurements and resume pending cases. Resume checks use the registered measurement plan and source directory, without hashing every source file again. Report-only edits can be rendered with cached measurements; changed model, settings or selected inputs require a fresh output directory.

For existing runs that lack authored continuations, rerun the same command with the same output directory. The report recovers their probabilities from the saved baseline and ablated logits.

The report retains the original investigation's Interpretation Costs. The extension's execution time and CUDA telemetry are recorded in its own `timeline.json` and `ablation/summary.json`.

| File | Evidence |
| --- | --- |
| `ablation/plan.json` | Source and model identity, seed, sampling rules, chosen cases, exact token IDs and positions |
| `ablation/cases/*/result.json` and `ablation/cases/*/logits.npz` | Per-case intervention records, output logits, target and control readouts |
| `ablation/summary.json` | Coverage by cohort and target/control medians |
| `status.json` and `timeline.json` | Run status and measured execution intervals |
| `report/index.html` and `report/paper-data.json` | Interactive report and renderer data |

Check the source checkpoint, precision and devices against the recorded run. Verify coverage against the plan and confirm that the saved target and controls each edit one unit at the recorded position. Recompute the report's medians from saved per-case results. In `report/index.html`, expand an example from each cohort and trace the prefix, product and probability readouts back to its measurement. These checks apply equally to a local Qwen3-1.7B run. The Colorado results refer to the Qwen3-32B neuron and its own source corpus.
