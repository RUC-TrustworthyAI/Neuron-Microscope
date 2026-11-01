# Method and implementation

The subject is a native dense-Qwen3 MLP neuron. A separate API model proposes explanations and experimental inputs. Python orchestrates structured requests, selects evidence, performs target-model measurements and renders the report. The entry point is `python -m language`; configuration is documented in [Workflow](workflow.md#run).

| Operation | Implementation | Origin of content |
|---|---|---|
| Scheduling, GPU lock and execution records | `language/workflow/run.py`, `language/workflow/state.py` | Fixed Python orchestration |
| Saved sample and evidence | `language/workflow/sampling.py`, `language/workflow/evidence.py` | Saved activations and captured contexts |
| Initial explanation and two revisions | `language/workflow/interpretation.py` | LLM sees all sampled measurements, then prediction feedback |
| Blind response predictions | `language/workflow/labels.py` | Independent requests with activations hidden |
| Prompts and output contracts | `language/workflow/prompts.py`, `language/workflow/contracts.py` | Defined requests and JSON structures |
| Activation search and controlled tests | `language/workflow/measure.py`, `language/workflow/run.py` | LLM proposes inputs; the target model supplies measurements |
| Attribution, family and interventions | `language/tools/relp.py`, `language/tools/paper_sections.py`, `language/workflow/interventions.py` | Measured target-model responses and derivatives |
| Blind maximum/minimum probes | `language/workflow/run.py`, `language/workflow/prompts.py` | Separate explanation-only writer and target-model scoring |
| Report analysis and independent reading | `language/workflow/revision.py` | LLM analyzes evidence against the frozen explanation |
| Report fields, plots and costs | `language/workflow/render.py`, `language/tools/report*.py` | Fixed rendering of structured prose and measured values |
| API requests, responses and usage | `language/workflow/client.py` | Complete per-request records and configured price estimates |

Measurements retain native BF16 gate, up and product channels, captured prefixes, explicit score positions and signed response poles. Corpus sample counts alone do not establish population semantic coverage. A coherent description, predictive agreement and a causal account each require evidence.

## Explanation and prediction

The explanation model receives every sampled position with its measured product value and contextual text. Additional evidence includes extreme tokens, document exemplars, quantile cases, token selectivity, peak statistics and direct logit effects. Gate and up channels are excluded from these requests. The configuration generator allows at most five rules per pole; the code falls back to three if a hand-written configuration omits this setting. Rule descriptions also form the report’s pole explanations. An unsupported pole may remain empty, and `No visible cue` covers contexts without a supported directional prediction.

An independent labeler receives the explanation, rules, full captured prefix through the anchor, and up to ten following tokens. Measured activations are hidden. The continuation supplies linguistic context; the causal target-model response at the anchor depends on its prefix. The labeler considers distant context and implicit relationships as well as nearby words.

Fixed code joins predictions to measurements. Two revisions each read the preceding explanation, all 500 measured contexts and focused error feedback. This feedback highlights sign reversals and positions above 50% of the corresponding corpus extreme assigned to No visible cue. Each revision is blind-labeled again, giving three versions. Among versions with directional accuracy >90%, code selects the highest coverage; otherwise it selects the highest accuracy. The selected explanation and its own predictions are frozen together before experimental design.

Activation spectrum and Inside the neuron share those final predictions and labeled positions. Spectrum bins and scatter coordinates use measured values; colors show predicted response groups. The samples supplied development feedback and therefore do not constitute a held-out evaluation. Independent written probes and controlled experiments test the frozen account on further inputs. See [sampling and missing-label handling](data.md#coordinates-and-sampling).

## Experiments and scoring

With `NEURON_MICROSCOPE_GPU_WORKER=1`, one resident target model per GPU group serves queued operations across runs. API calls can overlap with GPU work from other runs. Without the worker, the target model remains loaded for a run’s complete measurement stage under the group lock. Forward requests use `batch_size` and `batch_tokens`; separate groups can measure concurrently.

The generated configuration runs three Search rounds, requesting six regions and 16 texts per region. Measured feedback guides later rounds, and scoring uses the maximum or minimum native product over each full input. The separate Blind writer receives only the frozen claim and its rules, with no corpus examples or Search results, and writes six texts per supported pole. Blind also scores full-input extrema.

Four controlled tests use Search measurements to design ten conditions each. A test can target one pole with neutral comparisons. After measurement, the model improves the experimental design once, and code retains the stronger version of each test. Each condition stores its complete input separately from its short label. Tests score the replacement span, a common suffix after it or a specified token. Span scores retain the sign of the largest absolute response, with one scoring rule per contrast. The head figure selects the test with the highest positive response when positive rules exist, otherwise the strongest negative response; response range breaks ties.

The workflow also measures a natural and written attribution atlas, family responses, three multi-hop graphs, and matched interventions. The three graph roots are selected from the strongest positive search response, the strongest negative search response and the strongest absolute controlled response. These selections describe the measured cases; they do not represent an exhaustive circuit census.

## RelP and intervention scope

`language/tools/relp-source.json` pins the Transluce source revision and describes adaptations. Native HF forward values are retained while the backward pass freezes RMS scales and attention Q/K, uses the SiLU identity rule and the gated half rule. Local MLP-to-MLP contributions block intervening MLP outputs. The displayed graph additionally weights a local edge by its receiver's influence on the root.

These modified-derivative measurements provide attribution evidence. They do not give exact additive causal shares. Separate native zeroing and scaling measurements evaluate effects at explicit token positions. Family measurements and upstream/downstream tables retain their own selection and aggregation scope.

## Report analysis and semantic clusters

After measurements, the analysis request receives the frozen explanation, prediction feedback, measured search and control cases, blind tests, attribution summaries and interventions. It writes section prose, criticism and semantic groupings. The code preserves the frozen claim and response rules. Experimental contradictions remain in the analysis and criticism.

Semantic cluster memberships refer to measured atlas cases, including unresolved cases. Fixed code derives numerical summaries and keeps the underlying measurements. Numerical NMF routes remain separate: a shared numerical component alone does not establish a shared semantic topic or causal mechanism.

An independent reader assesses the account, supported branches and remaining patterns. The renderer then combines this prose with the measured data. Static Promotes/Suppresses values come from the input report's measured logit effects.

## Independent inputs and records

Interpretation requests contain no official neuron description or historical explanation. Reference snapshots are read only during comparison export, after the investigation is complete.

Each run saves its configuration, sampled evidence, explanation and prediction versions, complete API requests and outputs, usage estimates, measurements and final artifacts. Failed or incomplete operations remain recorded. Software tests check implementation behavior; scientific evidence comes from the saved measurements.
