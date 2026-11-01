# Project conventions

- Maintained Python code belongs in `vision-repro/src/vision_repro`; external model assets in `source/<model>/`; generated runs in unique `vision-repro/outputs/` directories.
- Execute pretrained optimization on a CUDA host in a dedicated environment. Do not change shared drivers, global packages or other users' jobs.
- Preserve explicit layer, channel, spatial aggregation and pre/post-nonlinearity definitions. Post-ReLU minima are not signed negative examples.
- Historical Lucent Inception5h is distinct from torchvision GoogLeNet and Goodfire retrained checkpoints. Record source revision and weight SHA-256.
- Each run records configuration, seed, provenance, versions, history, images and same-definition initial/final scores.
- Run targeted tests and a short batch-one pretrained GPU check after relevant numerical changes. Do not claim semantic reproduction from a smoke test.
- Never commit credentials, authorization links, datasets, model weights, machine-specific connection details or generated output archives.
- Keep full-generation campaigns resumable. Do not modify running campaign source or stop unrelated processes.
