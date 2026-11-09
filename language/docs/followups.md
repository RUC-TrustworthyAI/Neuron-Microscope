# Followup experiments from Criticism

Followups reproduce the public investigation flow and compact report format of
the [Colorado case](https://already-clean.goodfire.com/neuron-scientist/L55_N11703/).
Its internal prompts and scheduler were not published, so internal equivalence cannot be verified.
The explanation, blind labels and original Criticism stay frozen; followups investigate their counterexamples and doubts.

## Run and budget

Configurations from `language/examples/configure.py` enable:

```json
"followups": {
  "max_rounds": 2,
  "max_experiments": 6,
  "max_conditions": 36,
  "max_input_tokens": 1024
}
```
Run with `python -m language --config ... --output ...`. Omitting `followups` or setting
`max_rounds: 0` retains the baseline workflow. `--stop-after interpretation` stops before model experiments.

This implementation uses the Colorado case's observed counts as ceilings: at most six investigations
in round one and five in round two, also respecting a smaller `max_experiments`.
Use fewer or stop early when no useful question remains. `max_conditions` limits
distinct new inputs per investigation; several readouts share one native trace.
Saved graph groups are independent of this limit. Inputs exceeding the token
budget are unavailable rather than truncated; valid sibling investigations run.

## Investigation cycle

1. Start from upstream Criticism; optionally inspect saved evidence before design.
2. Register one falsifiable question, changed and fixed conditions, scored tokens
   and a textual prediction explaining which outcomes distinguish the competing
   explanations. Keep matched probes or a sweep in one investigation.
3. Execute the registered operations and retain actual measurements.
4. The result API receives complete condition text and measured readouts. It gives
   the decisive contrast, exceptions and tested scope. The verdict judges the
   tested working claim: `survived`, `narrowed`, `rejected` or `inconclusive`.
5. A short `suggested_revision` can change the affected local rule. The next round
   can test that revision or an unexpected observation using compact prior results.

Investigations cite source Criticism, prior investigations or measured evidence. Working claims
retain their parent and producing investigation. Missing evidence can be deferred with a reason.
Change matched prefixes and score at or after the change; suffix changes test earlier-token invariance.

## Evidence operations

Leave unused operation arrays empty. Saved inspection schedules no new forwards.

- `probes`: score exact new text or a saved `case_id`, with named readouts.
  Locations are the complete `input`, a literal `marker` with zero-based
  `occurrence` and optional `token_offset`, `token_position` (`start` only), or
  half-open `token_range` (`start`, `end`). Reductions are signed `max`, `min`,
  `absmax` or `single`; metrics are `product` and `peak_position`. Saved full-input
  and marker reads require a complete trace; partial traces support saved positions.
- `graph_queries`: select explicit `case_ids` or a `cohort_id` per group and a
  layer/neuron. Return presence, available rank/influence/activation statistics
  and scope. Optional `view: "census"` surveys observed units; `census_limit`
  limits the returned list to 1–128 (default 32), not the counted graph population.
  These are attribution observations; omitted nodes do not become zero activations.
- `resamples`: draw without replacement from sorted eligible saved cases using
  `sample_size` and `seed`, optionally filtering `group` and `activation_band`.
  Pair each isolated replay with its original product; retain pairs, `tolerance`,
  reproduced fraction and scoped `quantile` statistics. Isolation ends at the
  original anchor and uses its decoded visible token span. Unavailable spans
  cannot be replayed. Fresh no-cue probes can accompany the replay.
- `quantiles`: read a registered `quantile` from an existing weighted distribution,
  preserving its population scope. A development sample cannot stand for the corpus.

The archive comes from project measurements: native traces, saved spectrum contexts and
frozen groups, actual RelP graphs and available distributions.
Whole-corpus quantiles require the native BF16 histogram. Missing cohorts or
histograms stay unavailable; the Colorado reference is never measurement input.
New forwards retain token IDs, offsets and native `gate`, `up` (stored as `value`) and `product`, including the scoring token.

## Import saved evidence

Optional `followup_archive` is an absolute JSON path with this envelope:

```json
{"schema_version": 1, "identity": {"model": "...", "neuron": {}, "settings": {}}, "archive": {}}
```
Copy `model`, numeric layer/index `neuron` and the complete `settings` from the
configuration; identity must match. Include at least one evidence domain:

| Domain | Contents |
| --- | --- |
| `traces` | Case-ID map: `text`, ordered native `trace`, matching `selected`; retain token IDs, offsets and `isolation_text` when available. |
| `graphs` | Case-ID map: `nodes` with numeric `layer`/`neuron`; saved `rank`, `influence` and `activation` may be absent. |
| `cohorts` | Cohort-ID map of existing IDs, containing only traces or only graphs. |
| `cohort_descriptions` | Nonblank membership description for every imported cohort. |
| `distributions` | Distribution-ID map: finite `values`, positive integer `counts`, nonblank population `scope`. |

Imports retain source provenance, accept exact duplicates and reject conflicting IDs. Missing node statistics stay missing.
The importer reads only the explicit project-owned JSON path and does not discover reference reports.

## Report and resume

The unchanged Colorado UI shows round, verdict and challenge; expanding shows Prediction, Result and evidence. Its fields are
`round`, `attack`, `prediction`, `result`, `verdict`, `evidence` and
`suggested_revision`. Conditions and conclusions belong in the concise prose;
local revision state adds no second explanation panel or round-zero table.

`followups/context.json` retains the frozen sources and archive fingerprint.
Each round saves registration before execution, then measurements, API results
and claim versions; `summary.json` joins the rounds. API records are under
`api/followups/`. Full sweeps, traces, graph statistics and replay pairs stay on disk. Resume reuses completed work
with the same schema, configuration, source evidence and request inputs. Changed inputs or prompts require a new directory.
