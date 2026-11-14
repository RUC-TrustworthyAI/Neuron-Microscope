# Criticism-driven follow-up investigations

Follow-up investigation starts from a **completed** foundation investigation. It preserves the original explanation, predictions and Criticism, then investigates at most five questions in each of two rounds. It stops when the planner has no new distinguishable question, when all proposed questions repeat earlier ones, or when an entire round fails. It never pads a round to five.

## Protocol

Each question registers its original Criticism IDs, working hypothesis, rival, competing predictions and distinguishing observation before accessing new measurements. The planner can choose:

- **Review:** read identified examples, graphs or the baseline ablation summary, including full contexts from the preceding round. No new model forward is required.
- **Contrast:** specify literal prefixes, suffixes and replacement fragments. Python constructs every condition and scoring span; it never extracts replacements from model-written full sentences. One to three frames share two to six conditions. Every declared condition is retained, with the first as reference. Signed product and differences from that reference are measured without selecting a winning frame.
- **Corpus:** scan the complete saved activation archive on CPU. Two to four literal-query groups sample up to twelve matching documents each by seeded reservoir sampling, excluding documents shown during initial interpretation. Document selection never uses activation strength. The first literal match is scored with signed absolute maximum. Exact saved tokens, all three native channels, the match span and the causal prefix through the selected token are retained. Groups can overlap; empty groups and exclusions are recorded. These lexical samples do not establish semantic coverage or population prevalence.

The generation prompt includes the JSON schema before every call. The shared client uses JSON mode and at most one automatic contract correction; this is not provider-enforced strict JSON Schema. Semantic fields are structured before generation. Token limits and valid scoring positions are checked before GPU execution, without silent truncation or changing an experiment to make it fit.

Round two receives the **entire original Criticism**, frozen explanation, **every round-one question, specification, case outcome, assessment and failure**, and the accumulated proposed revisions. Each case retains its selected response, scored-region summary, scoring positions (lossless ranges), and a labeled input excerpt. Complete token traces and inputs remain in the archive and are available through review by case ID. Baseline graphs are referenced rather than repeatedly embedded in the history. The immediate assessment sees complete case text. No case outcome is replaced by just a final verdict or a shortlist of unresolved items.

Assessments record result, verdict, suggested revision and evidence IDs. Revisions remain a working ledger; they never overwrite the initial explanation or its blind labels. These adaptive investigations are development evidence, not held-out validation. The two reserved neurons remain separate and are not discovered or run by this command.

## Run when authorized

Use the existing Linux environment and credentials supplied through `NEURON_PAPER_API_KEY_FILE` (outside the repository) or `DEEPSEEK_API_KEY`. To run the foundation and follow-ups with one command:

```bash
python -u -m language --config workflow.local.json --output runs/example --followup
```

The command completes or resumes the foundation in `runs/example`, then starts follow-ups in the sibling directory `runs/example-followup`. The final report is `runs/example-followup/report/index.html`. Defaults are seed 42, 1,024 tokens per contrast input and up to 12 natural examples per query group. The foundation's explanation, predictions, report and completed status remain intact if a follow-up fails.

`--followup` requires a full `--config` investigation. It cannot be combined with `--stop-after interpretation`, `--ablate-from` or `--anatomy-from`; `--checkpoint-attribution` can be used alongside it. When `publish` is configured, successful follow-ups export the final report and update any existing export without changing the research files.

The standalone entry point remains available to select a different output directory or override the seed and limits:

```bash
python -m language.workflow.followups \
  --source runs/foundation/interpretations/L22_N6010 \
  --output runs/followups-L22_N6010 \
  --seed 42 --max-tokens 1024 --natural-per-group 12
```

The source and output must be separate, non-nested directories. Text measurements use one local model and batch size one; the default per-input budget is 1,024 tokens. Natural queries reuse saved activations and require no GPU. No reference blog explanations are supplied to the model.

No human inspection or adjustment is part of execution. A failed item is archived and does not discard successful items; all failures are visible to round two. If any item fails, final status is `complete_with_failures` and the CLI exits nonzero. A planner or infrastructure failure sets `failed`. Invalid outputs are never converted into successful scientific results. There are no transport retry loops or automatic changes to scientific parameters.

Repeat the identical command after interruption to reuse completed API and measurement checkpoints. With `--followup`, a completed foundation is reused before resuming the follow-up stage. Both entry points return a nonzero exit code when a follow-up fails. Failed item records remain terminal; use the standalone entry point with a new follow-up output directory for a new attempt. The manifest pins source files, configuration, options, implementation, model metadata and archive metadata. Large weight and activation arrays use size and modification time rather than full-file hashes. Changing these inputs rejects reuse.

## Artifacts and software checks

`frozen.json` preserves the baseline; `r*-i*/preregistered.json`, `evidence.json` and `record.json` retain specifications, raw measurements and decisions. `round-*.json` and `investigation.json` retain the full outcome ledger. `api/` contains the shared client's requests, responses and usage; credentials are not recorded. `status.json` reports completion or failure. The separate `report/` appends compact Criticism rows, preserving every other baseline report field, including Task2 and Task3.

```bash
PYTHONPATH=language/tools python -m unittest language.tests.test_followups -v
```

The tests use synthetic archives and fake API/model objects. They verify software behavior and are not scientific experiments or evidence of improved interpretation quality.
