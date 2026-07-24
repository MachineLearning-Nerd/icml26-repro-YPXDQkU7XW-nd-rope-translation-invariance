# nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding

This repository provides the core implementation of **nD-RoPE**, a generalized rotary position embedding for arbitrary-dimensional inputs.

nD-RoPE extends RoPE from 1D sequences to high-dimensional domains by treating positions and frequencies as unified n-dimensional vectors instead of decomposing them along independent coordinate axes.

<p align="center">
  <img src="figures/overview.png" width="92%">
</p>

<p align="center">
  <em>
  nD-RoPE models position-frequency interactions through unified n-dimensional wave vectors and constructs isotropic frequency directions using regular simplex geometry.
  </em>
</p>

---

## Overview

This repository evaluates nD-RoPE across representative architectures and tasks spanning images, videos, and point clouds.

| Task | Architecture | Base Repository |
|---|---|---|
| Image Classification | RoPE-ViT / DeiT | [facebookresearch/deit](https://github.com/facebookresearch/deit) |
| Video Recognition | TimeSformer | [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer) |
| Point Cloud Segmentation | Point Transformer | [qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers) |

---

## Key Idea

Conventional high-dimensional RoPE variants often apply rotary embeddings independently along each axis:

```math
\omega_x x + \omega_y y + \omega_z z
```

This axis-wise formulation introduces directional bias and weak cross-dimensional interaction.

nD-RoPE instead treats both positions and frequencies as unified n-dimensional vectors:

```math
e^{j \boldsymbol{\omega}^{\top} \mathbf{x}}
```

where `x` is the token coordinate and `ω` is an n-dimensional wave vector.

Wave vectors are constructed using a **multi-scale regular simplex design**, producing isotropic and non-axis-aligned frequency coverage.

---

## Dataset Preparation

Please download the required datasets from their official sources.

| Dataset | Task | Download Link |
|---|---|---|
| ImageNet-1K / ILSVRC 2012 | Image Classification | https://image-net.org/challenges/LSVRC/2012/index.php |
| Kinetics-400 | Video Recognition | https://github.com/cvdfoundation/kinetics-dataset |
| ModelNet40 | Point Cloud Segmentation | https://modelnet.cs.princeton.edu/ |

After downloading the datasets, update dataset paths in the corresponding config files or scripts.

- **RoPE-ViT / DeiT**: modify dataset paths in training and evaluation bash scripts.
- **TimeSformer**: modify dataset paths in YAML config files.
- **Point Transformer**: place datasets under `data/` or modify dataset paths in config files.

---

## 1. Image Classification: RoPE-ViT / DeiT

### Core Implementation

```text
models_v2_ndrope.py
```

### Training

```bash
bash run_ndrope.bash
```

### Evaluation

Evaluate without resolution extrapolation:

```bash
bash evaluate_ndrope.bash
```

Evaluate with YaRN:

```bash
bash evaluate_ndrope-yarn.bash
```

---

## 2. Video Recognition: TimeSformer

### Core Implementation

```text
timesformer/models/vit_ndrope.py
```

### Configuration

```text
configs/Kinetics/Timesformer_divST_8x32_224-ndrope.yaml
```


### Training

```bash
bash train_ndrope.bash
```

### Testing

Use the corresponding `_TEST.yaml` config file:

```bash
bash test_extrapolation-yarn.sh
```

---

## 3. Point Cloud Segmentation: Point Transformer

### Directory Structure

```text
PCT/
├── Point-Transformers-ndrope
└── Point-Transformers-ndrope-vector
```

### Core Implementation

```text
models/PT_ndrope/transformer.py
```

### Configuration

```text
config/partseg_ndrope.yaml
```

### Training

```bash
python train_partseg_ndrope.py
```

### Density Extrapolation Evaluation

```bash
bash test_partseg.bash
```

---

## Resolution and Density Extrapolation

<p align="center">
  <img src="figures/resolution_extrapolation_plots.png" width="92%">
</p>

<p align="center">
  <em>
  nD-RoPE improves extrapolation robustness across images, videos, and point clouds under large resolution and density shifts.
  </em>
</p>

---

## Citation

```bibtex
@inproceedings{
li2026ndrope,
title={nD-Ro{PE}: A Generalized Ro{PE} for n-Dimensional Position Embedding},
author={Boyang Li and Yulin Wu and Sizhe Xu and Nuoxian Huang and Zhonghang Yuan and Shangyi Guo and Shu Yang and Takahiro Yabe},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```