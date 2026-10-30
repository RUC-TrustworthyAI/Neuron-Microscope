# Input data

The pipeline takes **saved native activations** and local model weights as inputs. Activation archives can contain any number of tokens, provided they follow the format below.

## A target report package

The workflow configuration’s `report` path points to a target package containing:

| File | Contents |
|---|---|
| `index.html` | Activation-only report using the bundled renderer and JSON bundle |
| `report.json` | Identity, population statistics, selected positions, full saved documents and three-channel traces |
| `token-rankings.json` | Six token-selectivity tables, generated from saved activations |
| `dataset-index.json` | Opaque report case IDs mapped to canonical document/token positions |
| `effects.json` | Measured static logit effects used by Promotes/Suppresses |
| `peak-census.json`, `peak-census-negative.json` | Optional saved censuses for the corresponding report sections |

`report.json` includes `identity` (`id`, `layer`, `neuron`, `model`), `statistics`, `methods`, `examples`, `bands`, `scatter`, `points` and `documents`. Each point identifies its document, token position, and `[gate, value, product]`. Each document contains exact `token_ids`, decoded `tokens`, and aligned `gate`, `value`, `product` arrays. Cases are report-local identifiers; their names are not global token offsets.

The explanation model receives the sampled contexts with their measured gate, up and product values. Independent labeling requests hide those measurements. Prediction feedback supports one explanation refinement, followed by a second blind labeling pass. The refined explanation and rules are frozen before experiments. Official explanations and previous investigation conclusions are excluded from research inputs.

## Canonical activation archive

The bundled adapter is `tools/dataset_reader.py`. A dataset-local `reader.py` can implement the same interface and is treated as trusted executable input.

```text
archive/
  dataset.json
  neurons.json
  documents.parquet
  token_ids.npy
  indices/...
  part-0/
    shard-000000.npy
    histogram.npy
```

A manifest references paths instead of depending on those example names. Its fields used by the adapter are:

```json
{
  "complete": true,
  "tokens": 1024,
  "neurons": "neurons.json",
  "channels": ["gate", "value", "product"],
  "corpus": {"documents": "documents.parquet", "token_ids": "token_ids.npy"},
  "parts": [{
    "id": "part-0", "path": "part-0",
    "shards": [{"id": 0, "path": "shard-000000.npy", "token_start": 0, "token_count": 1024}],
    "indices": {"histogram": "histogram.npy"}
  }],
  "indices": {"k": 30, "extremes": "indices/extremes.npz", "statistics": "indices/statistics.parquet"}
}
```

- `neurons.json` is a list with `id`, `layer`, `neuron`, `part` and `part_column` per neuron.
- Document rows, sorted by `document_id` and corpus order, have `document_id`, `token_start`, `token_count`, `shard_id` and `shard_token_start`.
- Token IDs are a one-dimensional unsigned integer NPY array.
- Each activation shard is **neuron × token × channel**, stored as the raw BF16 bit patterns in `uint16` NPY arrays. These are not numeric uint16 activation values.
- Channel order is raw gate preactivation, raw up/value preactivation, and the captured native product at the input of `down_proj`.
- Histograms have shape `[part_neurons, 3, 65536]` and `uint64` counts, one bin for each BF16 bit pattern.
- Documents must retain the same causal input prefix used during collection. Document boundaries must align with shards for this adapter.

The scan uses `histogram`, `iter_neuron`, document rows and token IDs. The optional extremes/statistics methods use the additional index files. A custom adapter exposes `manifest`, `neurons`, `parts`, `locations`, `documents`, `starts`, `token_ids`, `document()`, `document_activations()`, `iter_neuron()`, `histogram()` and `close()`.

## Generate initial reports from an existing archive

`analysis/build_dataset_reports.py` builds initial reports from saved activations. It expects the original token-input sidecars (`block-NNNNNN.ids.u32.npy`, `.offsets.u64.npy`, `.documents.json`) and capture metadata in the archive manifest: `model`, `model_snapshot`, `capture`, `selection`, `documents` and `neuron_count`.

```bash
python analysis/build_dataset_reports.py \
  --dataset /path/to/archive --inputs /path/to/input-sidecars \
  --tokenizer /path/to/model/tokenizer.json \
  --existing /path/to/existing-reports --output /path/to/new-reports \
  --workers 2 --seed 42
```

Omit `--existing` if there are no reports to reuse. This is a CPU saved-data scan. Add token tables to copies with the separate saved-data command:

```bash
python analysis/add_token_tables.py --help
```

Alternatively the report builder accepts `--token-tables --token-cache DIR` to reuse the token-count/label cache in the same scan. The full argument list is available with `--help`.

## Coordinates and sampling

The measured activation is `SiLU(gate) * up`, taken from the native model tensor rather than reconstructed from rounded display values. Scatter point colours show blind predictions from a frozen explanation: orange for positive, blue for negative and grey for uncertainty. Stored records keep the prediction, measured activation and explanation version separately.

The workflow samples 500 positions from saved activations with seed 42. Each bin receives a largest-remainder quota proportional to `log10(full-corpus bin count)`, capped by its population; positions within the bin are sampled uniformly without replacement. Bins with zero log height receive no points. The explanation model sees the full sample with its measurements. For blind labeling, the predictor receives the full captured prefix through each anchor and up to ten following tokens, with measured responses hidden. Following tokens provide linguistic context and are outside the causal prefix at that anchor.

Activation spectrum and Inside the neuron share this sample pool. The initial explanation is labeled, refined once from prediction feedback, and labeled again. Both plots use the final predictions from the refined explanation, which is frozen before experiments. Each labeling pass runs two concurrent requests with up to 256 points each. Further rounds request missing predictions until fewer than 16 remain; these missing points are excluded from both plots. An explicit uncertain answer stays in No visible cue. The full-corpus histogram retains all token counts. The sample supports explanation development; it is neither a population density estimate nor a held-out evaluation.

All saved contexts used for model experiments preserve the captured prefix through the chosen anchor. The display can show a shorter window without changing the model input or scoring position.

## Corpora used in the comparisons

These are our saved evaluation inputs, counted with the Qwen3-32B tokenizer. Their labels refer to the following samples. Source links provide upstream data; the exact sampled texts and activation archives do not yet have a public download.

| Report label | Tokens | Source and selection |
|---|---:|---|
| SlimPajama | 100,000,000 | [DKYoon/SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B), revision `b5f90f419b7489cdba26fdbc8c022fcb5562f968`. Assign uniform random priorities to all 5,489,000 documents in its 48 train shards, seed `2026102901`; consume in priority order and trim only the final document to the budget. The sampling population is this public 6B subset. |
| FineWeb | 99,999,922 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Uniform document permutation from our saved 1B-token activation archive, seed `2026102905`, retaining 159,156 samples near the 100M boundary. Captured prefixes cut by a token limit were trimmed to the last detected sentence ending. Native activation bits were reused. |
| Common Pile | 100,000,067 | [common-pile/comma_v0.1_training_dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset), train. Complete documents taken from the beginning of our seeded, shuffled 500M-token pool until reaching 100M. This sample includes scientific text, code, books, patents, legal text, wikis and forums. |
| Diverse | 100,000,000 | Our source-quota mixture below. Complete records sampled from each component of our 500M pool, seed `2026102901`, with exact token-quota completion. Other-language records are interleaved by cumulative within-language token share. |
| Reasoning Mix | 100,000,000 | [EleutherAI/reasoning-mix](https://huggingface.co/datasets/EleutherAI/reasoning-mix), revision `325463c60e4d283679279277c1ee2f2aebe87e95`. Uniform random permutation of all 11,693,047 Parquet rows, seed `42`, without replacement. Each document contributes at most its first 4,096 tokens; the final document may be trimmed. The selected 99,724 documents comprise 81,143 `fineweb`, 18,448 `numina` and 133 `stratos` records; these are document counts, not token proportions. |

SlimPajama's sampled token proportions are Common Crawl 52.94%, C4 26.87%, arXiv 5.13%, GitHub 4.20%, books 3.93%, Wikipedia 3.78% and StackExchange 3.14% (rounded). No content filter or source quota was applied to that draw. Common Pile's largest sampled components are USPTO 34.33%, Stack v2 educational code 14.90%, peS2o 9.47%, PubMed 7.62% and StackExchange 5.24%; this sample has its own source balance.

### Diverse source quotas

Diverse is a coverage-oriented evaluation mixture assembled from public datasets. It contains 90M tokens of pretraining-style content and 10M tokens of reasoning, instruction and interaction content. These are our evaluation allocations; Qwen's training mixture is not reproduced here. A source's inclusion does not imply Qwen3 trained on that dataset.

| Component | Tokens | Public sources |
|---|---:|---|
| General web and web styles | 22.5M | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 18.5M; [C4](https://huggingface.co/datasets/allenai/c4) news 4M |
| Code | 18M | [Common Pile](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset), `stackv2-edu` |
| Mathematics | 9M | [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) 6M; [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) 3M |
| Scientific and professional documents | 9M | Common Pile: peS2o 3M, PubMed 2M, arXiv papers 1M, case law 0.4M, USGPO 0.4M, USPTO 0.2M; [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) 2M |
| Books and textbooks | 9M | Common Pile: Project Gutenberg 3M, pre-1929 books 2M, DOAB 2M, LibreTexts 1M, Pressbooks 1M |
| Forums and discussion | 9M | [Dolma](https://huggingface.co/datasets/allenai/dolma) Reddit 5M; Common Pile StackExchange 3M and Ubuntu IRC 1M |
| Multilingual web | 9M | [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2): Chinese 3M; French, Spanish, Portuguese, German, Russian, Arabic, Japanese, Korean, Hindi and Bengali 0.4M each; other languages 2M |
| Synthetic and annealing corpora | 4.5M | [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) Cosmopedia v2 2.5M; [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) math 1M; [OpenCoder annealing corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus) 1M |
| Long reasoning | 4M | [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) math 2M, code 1M, science 0.6M; [Bespoke-Stratos](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) 0.4M |
| Instructions | 2M | [Tulu 3 SFT](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) 1.2M; [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) constraints 0.4M; [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) 0.4M |
| Tool use | 1.6M | [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) 0.8M; [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) 0.8M |
| Multilingual dialogue | 1M | [Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset) |
| Retrieval-augmented answers | 1M | [Neural Bridge RAG](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) |
| Preference pairs | 0.4M | [Tulu 3 preference mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture), 0.2M chosen and 0.2M rejected |

The FineWeb allocation includes 15.5M general tokens plus 0.5M each for personal blogs, reviews, school exercises, sports/music/culture, local services and web essays. Synthetic content comes from existing datasets. The OpenCoder allocation retains its source records, but its natural/synthetic origin is not inferred solely from its folder name.

Raw documents can exceed a capture window. SlimPajama, Common Pile and Diverse were split into causal windows of at most 4,096 tokens for activation collection, with source positions retained. FineWeb and Reasoning Mix used captured document prefixes. Reports therefore describe the model's response to those saved inputs. Reproduction requires the corresponding input windows and tokenizer, not just a dataset name and token budget. Source datasets retain their own licenses and attribution requirements.
