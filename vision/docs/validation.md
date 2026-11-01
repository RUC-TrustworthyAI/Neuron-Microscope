# Release validation — 2026-10-30

| Check | Result |
| --- | --- |
| Python CPU suite | 65 passed; CUDA hidden during execution |
| RF export | Shallow/deep fixtures, all four buckets, boundary geometry, aligned heatmaps, static merge and corrupt-asset rejection |
| Preview compression | Source bytes preserved; heatmaps and activation positions unchanged |
| Viewer | Multirow atlases, embedded heatmaps, static/cloud fallback and quality-review placeholders passed |
| R2 API | Authentication, expiry, bounds, immutability, independent publication and reads passed |

This release validated export and viewer changes with the existing numerical optimizer.

## Previous pretrained GPU check — 2026-10-27

| Check | Result |
| --- | --- |
| CPU unit tests | 60 passed in 1.43 s |
| Frontend module syntax | Node syntax check passed |
| Model provenance check | Pinned Lucent commit and Inception5h SHA-256 match; no tracked source modifications |
| Pretrained CUDA smoke | One image, `mixed4c:447`, Positive, seed 0, 224 × 224, 8 updates |
| Initial / final evaluation | Gaussian-weighted post-ReLU: 0.0 → 855.375366 |
| CUDA allocation telemetry | 115.35 MiB peak allocated; 132 MiB peak reserved by this process's PyTorch allocator |
| Export smoke | 5,804-neuron catalog, one populated target, correct actual 8-step result and initial image |
| Wheel build | Successful; packaged defaults and HTML/JS/CSS confirmed present |

The GPU smoke completed on NVIDIA GB10. PyTorch reported device capability 12.1 against its advertised maximum of 12.0. Allocator telemetry measures this process's PyTorch allocations.

The eight-step smoke covers optimizer execution and record/export integrity. CPU tests cover independent-channel batching, signed targets, redirected gradients, color losses, queue coverage, export integrity and corrupt/traversing assets.
