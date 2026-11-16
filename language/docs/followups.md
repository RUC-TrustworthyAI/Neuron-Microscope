# Criticism-driven follow-up investigations

Follow-up investigation starts from a **completed** foundation investigation. It preserves the original explanation, predictions and Criticism, then investigates at most five questions in each of two rounds. It stops when the planner has no new distinguishable question, when all proposed questions repeat earlier ones, or when an entire round fails. It never pads a round to five.

## Protocol

Each question registers its original Criticism IDs, working hypothesis, rival, competing predictions and distinguishing observation before accessing new measurements. The planner can choose:

- **Review:** read identified examples, graphs or the baseline ablation summary, including full contexts from the preceding round. No new model forward is required.
- **Contrast:** specify literal prefixes, suffixes and replacement fragments. Python constructs every condition and scoring span; it never extracts replacements from model-written full sentences. One to three frames share two to six conditions. Every declared condition is retained, with the first as reference. Signed product and differences from that reference are measured without selecting a winning frame.
- **Corpus:** scan the complete saved activation archive on CPU. Two to four literal-query groups sample up to twelve matching documents each by seeded reservoir sampling, excluding documents shown during initial interpretation. Document selection never uses activation strength. The first literal match is scored with signed absolute maximum. Exact saved tokens, all three native channels, the match span and the causal prefix through the selected token are retained. Groups can overlap; empty groups and exclusions are recorded. These lexical samples do not establish semantic coverage or population prevalence.

The generation prompt includes the JSON schema before every call. The shared client uses JSON mode and at most one automatic contract correction; this is not provider-enforced strict JSON Schema. Semantic fields are structured before generation. Token limits and valid scoring positions are checked before GPU execution, without silent truncation or changing an experiment to make it fit.

Round two receives the **entire original Criticism**, frozen explanation, **every round-one question, specification, case outcome, assessment and failure**, and the accumulated proposed revisions. Each case retains its selected response, scored-region summary, scoring positions (lossless ranges), and a labeled input excerpt. Complete token traces and inputs remain in the archive and are available through review by case ID. Baseline graphs are referenced rather than repeatedly embedded in the history. The immediate assessment sees complete case text. No case outcome is replaced by just a final verdict or a shortlist of unresolved items.

Assessments record result, verdict, suggested revision and evidence IDs. Revisions remain a working ledger; they never overwrite the initial explanation or its blind labels. These adaptive investigations are development evidence, not held-out validation. Only the configured neuron is investigated.

## Run when authorized

Use the existing Linux environment and credentials supplied through `NEURON_PAPER_API_KEY_FILE` (outside the repository) or `DEEPSEEK_API_KEY`. To run the foundation and follow-ups with one command:

```bash
python -u -m language --config workflow.local.json --output runs/example
```

The workflow completes or resumes its foundation in `runs/example`, then runs follow-ups under the same execution lock. This is also the default for Python calls to `language.workflow.run.run`. Follow-up records remain in `runs/example/followup/` so existing checkpoints remain reusable. The root `status.json` tracks follow-up progress (`phase=followup`, with step, round and investigation) and reaches `complete` only after the final report is ready. It retains a snapshot of the original report and updates `runs/example/report/index.html` and `paper-data.json` with appended Criticism and combined cost totals. Defaults are seed 42, 1,024 tokens per contrast input and up to 12 natural examples per query group. The frozen explanation, predictions and original Criticism remain unchanged. Item failures are retained in the report, set workflow status to `complete_with_failures` and exit nonzero; fatal errors set workflow status to `failed`.

Follow-ups are enabled by default for full `--config` investigations; append `--no-followup` to disable them. Interpretation-only and replay modes skip them automatically. Explicit `--followup` cannot be combined with `--stop-after interpretation`, `--ablate-from` or `--anatomy-from`; `--checkpoint-attribution` can be used alongside it. When `publish` is configured, successful follow-ups export the updated report and replace any existing export. Failed follow-ups are not published.

The standalone entry point remains available to select a different output directory or override the seed and limits:

```bash
python -m language.workflow.followups \
  --source runs/foundation/interpretations/L22_N6010 \
  --output runs/followups-L22_N6010 \
  --seed 42 --max-tokens 1024 --natural-per-group 12
```

For the standalone entry point, source and output must be separate, non-nested directories; its report is written under the specified output and does not update the source report. Text measurements use one local model and batch size one; the default per-input budget is 1,024 tokens. Natural queries reuse saved activations and require no GPU. No reference blog explanations are supplied to the model.

No human inspection or adjustment is part of execution. A failed item is archived and does not discard successful items; all failures are visible to round two. If any item fails, final status is `complete_with_failures` and the CLI exits nonzero. A planner or infrastructure failure sets `failed`. Invalid outputs are never converted into successful scientific results. There are no transport retry loops or automatic changes to scientific parameters.

Repeat the identical command after interruption to reuse completed API and measurement checkpoints. When follow-ups are enabled, a completed foundation is reused before resuming the follow-up stage. Both entry points return a nonzero exit code when a follow-up fails. Failed item records remain terminal. A new attempt through the standalone entry point requires an unaugmented foundation report and a new follow-up output directory; an already augmented source report is rejected to prevent duplicate additions. Resume checks compare source-content digests, options, model metadata and archive metadata. Large weight and activation arrays use size and modification time rather than full-file hashes. Source-code hashes and obsolete manifest fields are ignored, so code refactoring alone does not invalidate completed checkpoints; changed scientific inputs still reject reuse.

Existing sibling directories such as `runs/example-followup` are not migrated automatically. The default layout keeps all new records inside `runs/example`. The [walkthrough exporter](walkthrough.md#export-another-completed-run) reads its updated report directly, so no separate report replacement is needed.

## Artifacts and software checks

Under `runs/example/followup/` (or the standalone output), `frozen.json` preserves the baseline. Each investigation saves its specification, measurements and decision. `round-*.json` and `investigation.json` retain all outcomes, and `status.json` reports follow-up completion or failure. The follow-up report appends compact Criticism rows. For full workflow runs, those rows and the foundation measurements feed the synthesized Claim in the final report.

```bash
PYTHONPATH=language/tools python -m unittest language.tests.test_followups language.tests.test_followup_resume -v
```

The tests use synthetic archives and fake API/model objects. They verify software behavior and are not scientific experiments or evidence of improved interpretation quality.
