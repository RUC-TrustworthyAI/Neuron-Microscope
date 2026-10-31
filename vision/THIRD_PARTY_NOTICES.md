# Sources and third-party assets

The vision implementation and bundled optimization figures were imported from [RUC-TrustworthyAI/Vision-Neuron-Microscope](https://github.com/RUC-TrustworthyAI/Vision-Neuron-Microscope/tree/72ea2d06094473d2d4368eb79338f5a428ffd636/vision), revision `72ea2d06094473d2d4368eb79338f5a428ffd636` on branch `merge`. Attribution remains with the upstream contributors. The root MIT license covers the LLM implementation; this directory retains the upstream licensing status below.

- Method inspiration and external comparison: [Neural Networks Are Already Clean / Nick & Goodfire](https://already-clean.goodfire.com/) and their [Microscope](https://already-clean.goodfire.com/microscope/).
- Model implementation: [greentfrapp/lucent](https://github.com/greentfrapp/lucent), pinned in `source/inceptionv1/manifest.json`.
- Historical model weights: [ProGamerGov/pytorch-old-tensorflow-models](https://github.com/ProGamerGov/pytorch-old-tensorflow-models), exact URL and SHA-256 in the manifest.
- Feature-visualization background: [tensorflow/lucid](https://github.com/tensorflow/lucid).
- Natural-image data: [ImageNet](https://image-net.org/), acquired separately under its applicable access and usage terms.

This module is an independent research implementation, not an official release by the above projects. It does not distribute their checkpoints, source checkouts, reference PNGs or ImageNet images. Optional reference URLs remain external; exporting a viewer does not grant redistribution rights to external or dataset assets. Review their respective terms before redistributing a result bundle.

Python and browser dependencies retain their own licenses. An open-source license for the upstream vision project's original code has not yet been selected; public availability alone does not grant a new license. Do not infer that upstream licenses automatically cover this project's original code.
