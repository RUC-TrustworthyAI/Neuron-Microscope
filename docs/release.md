# Source-only release

The scientific framework is synchronized with the 2026-10-26 semantic-cluster source snapshot. This is a separate publication checkout; active research run directories and their frozen source are not changed by preparing this repository.

Included current changes:

- Explicit semantic cluster membership over measured natural-atlas graph IDs, separate from unchanged NMF routes.
- Full-member evidence access for experiment and revision agents; unassigned measured contexts remain visible.
- Cluster definitions survive revision when omitted from a later annotation update.
- Scan annotation checkpoints can be resumed without discarding completed batches.
- Shared 128-token display context before the first highlighted token for exemplars and cluster examples.
- Optional local original/report comparison at publication, including section and subsection alignment, linked scrolling and a resizable divider.
- Cost aggregation can include saved additional API ledgers from a resumed run.

The portable host configuration is retained from the earlier source packaging: paths, tokenizer, Docker image, investigator model and tariff are explicit settings; credentials come from the host environment or a local file. The dataset reader and initial report builders are included, so runtime imports do not depend on a private research checkout. The Dockerfile uses public packages rather than a machine-local base image.

The initial commit contains no earlier repository history, model files, activation arrays, generated reports, original neuron-page snapshots, API records, or measured-result images. The offline demo creates synthetic data only when run. Scientific defaults and the three skill instructions are retained; this packaging adds no agent stage, scientific acceptance gate, automatic retry, or new measurement rule.

The semantic-cluster revision is an implemented workflow under evaluation. Its presence in this repository does not imply that a new scientific run has finished or that its interpretations have been independently validated.
