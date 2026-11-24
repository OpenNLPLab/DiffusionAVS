# Third-party code

The model and training utilities are adapted from the authors' MMMI_AVS code.

- AVSBench: https://github.com/OpenNLPLab/AVSBench — dataset loading, TPAVI, and segmentation utilities.
- PVT: https://github.com/whai362/PVT — PVT-v2 encoder in `lib/model/encoder/pvt.py`.
- torchvggish: https://github.com/harritaylor/torchvggish — audio encoder and preprocessing in `lib/torchvggish/`.
- TensorFlow Authors: VGGish preprocessing files retain their Apache-2.0 copyright notices.
- DiffusionSOD: reference for vector diffusion and latent fusion; the provenance supplied with the authors' reference should be checked before public distribution.

Original copyright notices are preserved. Dataset and pretrained-model weights are not included in this source repository and retain their respective terms.

This repository is a publication draft. A project-wide license has not yet been assigned; upstream redistribution terms and the license for the authors' contributions must be finalized before public release.
