# Release validation — 2026-10-30

| Check | Result |
| --- | --- |
| Python CPU suite | 65 passed; CUDA hidden during execution |
| RF export | Shallow/deep fixtures, all four buckets, boundary geometry, aligned heatmaps, static merge and corrupt-asset rejection |
| Preview compression | Source bytes preserved; heatmaps and activation positions unchanged |
| Viewer | Multirow atlases, embedded heatmaps, static/cloud fallback and quality-review placeholders passed |
| R2 API | Authentication, expiry, bounds, immutability, independent publication and reads passed |

The numerical optimizer is unchanged in this release. No new GPU optimization was run.

## Previous pretrained GPU check — 2026-10-27


Validation used a separate copy of this repository. The existing full-generation campaign and its environment installation were not modified.

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

The GPU smoke ran on NVIDIA GB10. The existing PyTorch build warned that device capability 12.1 exceeds its advertised maximum of 12.0; this particular execution completed successfully. No driver or framework replacement was attempted. This is not a general compatibility guarantee for every CUDA operation or a performance benchmark. Allocator telemetry excludes other processes and CUDA context/system allocations.

The eight-step smoke validates execution and record/export integrity, not visual quality or semantic equivalence to Nick / Goodfire. It did not rerun the ImageNet scan or the complete 29,020-request campaign. CPU tests cover independent-channel batching, signed targets, redirected gradients, color losses, queue coverage, export integrity and corrupt/traversing assets. CI on GitHub has not yet been run as part of this local validation.
