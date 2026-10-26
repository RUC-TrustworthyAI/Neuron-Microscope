# Evidence API

Run `scripts/read_evidence.py` relative to this skill directory. With no arguments it returns the evidence index for the assigned neuron. The host reads the target-only report package mounted at `/inputs/report`; the full activation archive and other neurons are not mounted.

Options:

| Option | Meaning |
|---|---|
| `--section` | `index`, `distribution`, `contexts`, `token_selectivity`, `peak_census`, `gate_value`, `output_effects`, `anatomy_scan`, or `all` |
| `--group` | One context group listed by the index |
| `--ids` | Specific case IDs |
| `--offset` | First context in the filtered list; default 0 |
| `--limit` | Contexts per page; default 4 |

A context page returns `total`, `offset`, `next_offset`, `trace_columns` and `cases`. Pagination retains the full observed prefix, anchor, local gate/value/product trace and following prediction target. It does not shorten a context.

The anatomy_scan section returns the prepared scan's semantic groups, clusters, interval allocation and paginated full-prefix cases. Use --limit 0 for definitions and counts, then --group, --ids or --offset/--limit for contexts. These points are additional observed evidence, not a previous neuron explanation.

The six token selectivity tables use Top20 rows and occurrence count at least 50. Thresholds are fractions of the signed corpus maximum/minimum. Positive, negative and absolute peak censuses are separate. The gate_value response supplies numeric rows at data.scatter, ordered by data.scatter_columns (gate, value, product). In raw report.json, scatter is a list of case IDs; their numeric rows are points[case_id].values. Output effects contain the existing static projections and native ablations.
