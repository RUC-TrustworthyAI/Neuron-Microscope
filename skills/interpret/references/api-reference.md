# Evidence API

Run `scripts/read_evidence.py` relative to this skill directory. With no arguments it returns the evidence index for the assigned neuron. The host reads the target-only report package mounted at `/inputs/report`; the full activation archive and other neurons are not mounted.

Options:

| Option | Meaning |
|---|---|
| `--section` | `index`, `distribution`, `contexts`, `token_selectivity`, `peak_census`, `gate_value`, `output_effects`, or `all` |
| `--group` | One context group listed by the index |
| `--ids` | Specific case IDs |
| `--offset` | First context in the filtered list; default 0 |
| `--limit` | Contexts per page; default 4 |

A context page returns `total`, `offset`, `next_offset`, `trace_columns` and `cases`. Pagination retains the full observed prefix, anchor, local gate/value/product trace and following prediction target. It does not shorten a context.


The six token selectivity tables use Top20 rows and occurrence count at least 50. Thresholds are fractions of the signed corpus maximum/minimum. Positive, negative and absolute peak censuses are separate. The gate_value response supplies numeric rows at data.scatter, ordered by data.scatter_columns (gate, value, product). In raw report.json, scatter is a list of case IDs; their numeric rows are points[case_id].values. Output effects contain the existing static projections and native ablations.

Claim prose uses approximate English word targets: lede 60–85, one_paragraph 150–180, negative_pole 80–120, what_would_change_my_mind 40–60. The latter three fields appear as the three paragraphs of The claim in detail; supporting evidence remains available separately.

Submit response_rules alongside claim: at most three positive groups, at most three negative groups, and exactly one uncertain group with id no-visible-cue and name No visible cue. Each rule has id, name, description and predicted_pole (`positive`, `negative` or `uncertain`). The description states the textual conditions and anchor location under which that prediction follows from the claim. Use No visible cue whenever the direction cannot be distinguished. Submission immediately runs per-sample labeling; the saved predictions and measurements are available to the revision agent.
