# Method and implementation

The subject is a native dense-Qwen3 MLP neuron. A separate API model proposes explanations and experimental inputs. Python orchestrates structured requests, selects evidence, performs target-model measurements and renders the report. The entry point is `python -m workflow.run`; configuration is documented in [Workflow](workflow.md#run).

| Operation | Implementation | Origin of content |
|---|---|---|
| Scheduling, GPU lock and execution records | `workflow/run.py`, `workflow/state.py` | Fixed Python orchestration |
| Saved sample and evidence | `workflow/sampling.py`, `workflow/evidence.py` | Saved activations and captured contexts |
| Initial explanation and one refinement | `workflow/interpretation.py` | LLM sees all sampled measurements, then prediction feedback |
| Blind response predictions | `workflow/labels.py` | Independent requests with activations hidden |
| Prompts and output contracts | `workflow/prompts.py`, `workflow/contracts.py` | Defined requests and JSON structures |
| Activation search and controlled tests | `workflow/measure.py`, `workflow/run.py` | LLM proposes inputs; the target model supplies measurements |
| Attribution, family and interventions | `tools/relp.py`, `tools/paper_sections.py`, `workflow/interventions.py` | Measured target-model responses and derivatives |
| Blind maximum/minimum probes | `workflow/run.py`, `workflow/prompts.py` | Separate explanation-only writer and target-model scoring |
| Report analysis and independent reading | `workflow/revision.py` | LLM analyzes evidence against the frozen explanation |
| Report fields, plots and costs | `workflow/render.py`, `tools/report*.py` | Fixed rendering of structured prose and measured values |
| API requests, responses and usage | `workflow/client.py` | Complete per-request records and configured price estimates |

Measurements retain native BF16 gate, up and product channels, captured prefixes, explicit score positions and signed response poles. Corpus sample counts alone do not establish population semantic coverage. A coherent description, predictive agreement and a causal account each require evidence.

## Explanation and prediction

The explanation model receives every sampled position with its measured gate, up and product values and contextual text. It proposes a claim and, by default, at most three positive and three negative response rules. Each rule describes when the explanation predicts a direction. `No visible cue` is added for contexts where neither direction follows clearly from the explanation.

An independent labeler receives the explanation, rules, full captured prefix through the anchor, and up to ten following tokens. Measured activations are hidden. The continuation supplies linguistic context; the causal target-model response at the anchor depends on its prefix. The labeler considers distant context and implicit relationships as well as nearby words.

Fixed code joins these predictions to measurements. The explanation model uses the measured contexts and prediction feedback to refine the explanation and rules once. A second blind labeling pass applies that refined version to the same sample pool. The explanation, rules and final predictions are frozen before experimental design.

Activation spectrum and Inside the neuron share those final predictions and labeled positions. Spectrum bins and scatter coordinates use measured values; colors show predicted response groups. The samples supplied development feedback and therefore do not constitute a held-out evaluation. Independent written probes and controlled experiments test the frozen account on further inputs. See [sampling and missing-label handling](data.md#coordinates-and-sampling).

## Experiments and scoring

The target model is loaded once per measurement stage and retained through search, controlled tests, attribution and interventions. Forward requests are batched according to `batch_size` and `batch_tokens`. Runs using the same physical GPU group share one lock; separate groups can measure different neurons concurrently.

Activation search uses measured feedback to propose further texts, scoring the maximum or minimum native product over each full input. The separate blind writer receives only the frozen claim and writes positive and negative probes without corpus examples or search results. These probes also use full-input maxima and minima.

Controlled tests use the search measurements to design ten conditions covering positive, negative and neutral cases. Each condition stores its complete input separately from its short display label. Tests score the changed span, a common suffix after it, or a specified token. Span scores retain the sign of the largest absolute response. The same scoring rule applies across a contrast. The headline selects a common-frame test with the strongest coverage of both observed corpus poles.

The workflow also measures a natural and written attribution atlas, family responses, three multi-hop graphs, and matched interventions. The three graph roots are selected from the strongest positive search response, the strongest negative search response and the strongest absolute controlled response. These selections describe the measured cases; they do not represent an exhaustive circuit census.

## RelP and intervention scope

`tools/relp-source.json` pins the Transluce source revision and describes adaptations. Native HF forward values are retained while the backward pass freezes RMS scales and attention Q/K, uses the SiLU identity rule and the gated half rule. Local MLP-to-MLP contributions block intervening MLP outputs. The displayed graph additionally weights a local edge by its receiver's influence on the root.

These modified-derivative measurements provide attribution evidence. They do not give exact additive causal shares. Separate native zeroing and scaling measurements evaluate effects at explicit token positions. Family measurements and upstream/downstream tables retain their own selection and aggregation scope.

## Report analysis and semantic clusters

After measurements, the analysis request receives the frozen explanation, prediction feedback, measured search and control cases, blind tests, attribution summaries and interventions. It writes section prose, criticism and semantic groupings. The code preserves the frozen claim and response rules. Experimental contradictions remain in the analysis and criticism.

Semantic cluster memberships refer to measured atlas cases, including unresolved cases. Fixed code derives numerical summaries and keeps the underlying measurements. Numerical NMF routes remain separate: a shared numerical component alone does not establish a shared semantic topic or causal mechanism.

An independent reader assesses the account, supported branches and remaining patterns. The renderer then combines this prose with the measured data. Static Promotes/Suppresses values come from the input report's measured logit effects.

## Independent inputs and records

Interpretation requests contain no official neuron description or historical explanation. Reference snapshots are read only during comparison export, after the investigation is complete.

Each run saves its configuration, sampled evidence, explanation and prediction versions, complete API requests and outputs, usage estimates, measurements and final artifacts. Failed or incomplete operations remain recorded. Software tests check implementation behavior; scientific evidence comes from the saved measurements.
