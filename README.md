# DiffusionAVS · TIP 2025

**Accepted by IEEE Transactions on Image Processing (TIP), 2025.**

**Contrastive Conditional Latent Diffusion for Audio-visual Segmentation**

Yuxin Mao, Jing Zhang, Mochu Xiang, Yunqiu Lv, Dong Li, Yiran Zhong, and Yuchao Dai

[[IEEE TIP Paper]](https://doi.org/10.1109/TIP.2025.3580269) [[arXiv]](https://arxiv.org/abs/2307.16579)

DiffusionAVS combines audio-visual features with a conditional diffusion model in a compact latent space. A ground-truth encoder represents segmentation masks, while an audio-visual encoder provides the condition for denoising. The resulting latent representation is fused with multi-scale TPAVI features to predict the sounding objects. Contrastive learning aligns the conditional and ground-truth representations.

This repository provides the code for DiffusionAVS on **MS3 and S4**, including training and evaluation. Both tasks share the same architecture.

## Results

Results on AVSBench with the PVT-v2-B5 backbone. Both metrics are measured using the same checkpoint.

| Task | mIoU (%) | F |
|---|---:|---:|
| MS3 | 59.46 | 0.7198 |
| S4 | 81.90 | 0.9047 |

These results use seed 1234. Results may vary with the training seed and environment.

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/maoyuxin/DiffusionAVS):

| Task | Weights | Configuration |
|---|---|---|
| MS3 | [best.pth](https://huggingface.co/maoyuxin/DiffusionAVS/resolve/main/MS3/best.pth) | [config.json](https://huggingface.co/maoyuxin/DiffusionAVS/resolve/main/MS3/config.json) |
| S4 | [best.pth](https://huggingface.co/maoyuxin/DiffusionAVS/resolve/main/S4/best.pth) | [config.json](https://huggingface.co/maoyuxin/DiffusionAVS/resolve/main/S4/config.json) |

## Installation

Use Python 3.11 and a CUDA-enabled PyTorch installation. Our experiments use PyTorch 2.6 and CUDA 12.4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and pretrained models

Download the AVSBench-object data following the [AVSBench instructions](https://github.com/OpenNLPLab/AVSBench). We use extracted video frames, ground-truth masks, and the provided audio log-mel features:

```text
data/AVSBench-object/
├── Multi-sources/
│   ├── ms3_meta_data.csv
│   └── ms3_data/
│       ├── visual_frames/
│       ├── gt_masks/
│       └── audio_log_mel/
└── Single-source/
    ├── s4_meta_data.csv
    └── s4_data/
        ├── visual_frames/
        ├── gt_masks/
        └── audio_log_mel/
```

Place the [PVT-v2-B5](https://github.com/whai362/PVT) and [VGGish](https://github.com/harritaylor/torchvggish) pretrained weights in:

```text
pretrained/
├── pvt_v2_b5.pth
└── vggish-10086976.pth
```

Alternatively, pass `--data-root /path/to/AVSBench-object` and `--pretrained-root /path/to/pretrained` to the commands below.

## Training

```bash
# MS3: 30 epochs
CUDA_VISIBLE_DEVICES=0 python run.py --task MS3 --output runs/ms3

# S4: 15 epochs
CUDA_VISIBLE_DEVICES=0 python run.py --task S4 --output runs/s4
```

Both tasks use Adam with learning rate `1e-4`, batch size 4 clips, and a learning-rate decay of 0.5 every 12 epochs. MS3 uses random horizontal flip and affine augmentation; S4 uses horizontal flip only. S4 applies segmentation supervision to the first frame and repeats its mask for the five diffusion targets in each clip.

The latent dimension is 24, with 20 diffusion steps and 10 DDIM sampling steps. We use VP parameterization with an **epsilon-MSE objective**. Training combines segmentation, semantic, diffusion, and contrastive losses, with semantic, diffusion, and contrastive weights of 1, 1, and 0.1.

Training uses the training split and selects `best.pth` by validation mIoU. Checkpoints, configuration, and training metrics are saved in the output directory.

## Evaluation

The `--evaluate` command evaluates a checkpoint on the test split.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --evaluate runs/ms3/best.pth
CUDA_VISIBLE_DEVICES=0 python run.py --evaluate runs/s4/best.pth
```

For the released weights, place `best.pth` in `checkpoints/MS3/` or `checkpoints/S4/`, alongside the supplied `config.json`, and use that path with `--evaluate`. The task and model settings are loaded from the checkpoint's configuration. Data and pretrained-model locations can still be set with the path arguments above.

## Citation

```bibtex
@article{mao2025diffusionavs,
  title={Contrastive Conditional Latent Diffusion for Audio-Visual Segmentation},
  author={Mao, Yuxin and Zhang, Jing and Xiang, Mochu and Lv, Yunqiu and Li, Dong and Zhong, Yiran and Dai, Yuchao},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  volume={34},
  pages={4108--4119},
  doi={10.1109/TIP.2025.3580269}
}
```

## Acknowledgements

This implementation builds on our [MMVAE-AVS](https://github.com/OpenNLPLab/MMVAE-AVS) code and draws on [AVSBench](https://github.com/OpenNLPLab/AVSBench), [PVT](https://github.com/whai362/PVT), and [torchvggish](https://github.com/harritaylor/torchvggish). We thank their authors for sharing their work. Third-party code retains its original notices; see `THIRD_PARTY_NOTICES.md`.
