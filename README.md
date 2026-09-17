# Adjoint Inversion Reveals Holographic Superposition and Destructive Interference in CNN Classiffers



---

## Overview

This repository contains the official PyTorch implementation for our NeurIPS paper, including:

- **Pretrained LAC weights** for `convnext_base` trained on ImageNet (10 000 steps)
- **Training script** (`imagenet.py`) for the Dual Manifold Framework
- **Visualization scripts** (`stage_vis.py`, `channel_vis.py`) for class attribution and per-channel spatial basis analysis
- **Pre-generated visualization PDFs** (`class_attribution_posneg.pdf`, `channel_vis_stage3_convnext_base.pdf`)

The **Dual Manifold Framework** is an architecture-agnostic CNN feature inversion method that decomposes internal representations into an orthogonal, stage-wise *Semantic Inversion Spectrum*. A lightweight **LAC (Layer-wise Adaptive Correction) Refiner** is trained at each stage boundary to map VJP signals back to interpretable pixel-space reconstructions — analytically, without iterative optimization at test time.

---

## Key Contributions

1. **Architecture-agnostic feature inversion** via timm, supporting ResNet, DenseNet, ConvNeXt, and other standard CNN families with zero code changes.
2. **LAC Refiner** — a learnable GroupNorm-based module at each stage boundary that corrects vector-Jacobian products, enabling clean per-stage spatial basis extraction.
3. **Class Attribution via Cancellation Effect** — decomposes the classifier gradient into positive and negative channel contributions, visualizing what each class excites vs. suppresses.
4. **Per-channel spatial basis visualization** — for any selected stage, renders the full set of energy-sorted spatial basis images `Ṽ_{l,c}` as a multi-page PDF.

---

## Pretrained Weights

| Checkpoint | Backbone | Dataset | Steps | Download |
|:-----------|:---------|:--------|------:|:---------|
| `imagenet_lac_convnext_base_step010000.pth` | ConvNeXt-Base | ImageNet | 10 000 | Included in this repo |

The checkpoint stores only the LAC parameters (not the frozen backbone):

```python
{
    'lac':      model.lac.state_dict(),       # stage-boundary refiners
    'lac_stem': model.lac_stem.state_dict(),  # pixel-boundary refiner
}
```

---

## Visualization Outputs

Two pre-generated PDFs are included:

| File | Description |
|:-----|:------------|
| `channel_vis_stage3_convnext_base.pdf` | Per-channel spatial bases `Ṽ_{3,c}` for all 1024 channels of ConvNeXt-Base Stage 3, sorted by energy. Page 1 shows the original image, full stage reconstruction, and energy distribution; subsequent pages show 10×10 channel grids. |
| `class_attribution_posneg.pdf` | Class attribution for 5 Stanford-Dogs samples. Columns: original image → stage inversions (Stage 0–3) → combined class reconstruction → positive-α channels → negative-α channels. |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# Create a conda environment (Python 3.9+)
conda create -n dmf python=3.9 -y
conda activate dmf

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install timm matplotlib numpy scipy Pillow
```

**Tested with:** Python 3.9, PyTorch 2.x, timm 0.9+, CUDA 11.8.

---

## Usage

### 1. Training the LAC Refiners

```bash
python imagenet.py
```

Key configuration (edit inside `imagenet.py`):

```python
MODEL_NAME   = 'convnext_base'   # any timm CNN backbone
WEIGHTS_PATH = None              # None → use timm pretrained; or path to custom .pth
BATCH_SIZE   = 20
LR           = 1e-3
TOTAL_STEPS  = 10000
TOP_K_TRAIN  = 8                 # Monte-Carlo channels per step
DEVICE       = 'cuda:0'
```

The script saves a checkpoint every 1 000 steps as:

```
imagenet_lac_{MODEL_NAME}_step{STEP:06d}.pth
```

Visualization snapshots are written every 500 steps to `./vis_results_{MODEL_NAME}_imagenet/`.

> **ImageNet path:** Update `data_dir` inside `get_imagenet_dataloaders()` to point to your local ImageNet root (expected layout: `{data_dir}/train/` and `{data_dir}/val/` in `ImageFolder` format).

---

### 2. Class Attribution Visualization (`stage_vis.py`)

Generates a multi-page PDF showing the stage-wise Semantic Inversion Spectrum and class attribution (positive / negative channels) for a set of test images.

```bash
python stage_vis.py
```

Configuration block at the bottom of the file:

```python
MODEL_NAME   = 'convnext_base'
WEIGHTS_PATH = 'weights/convnext_base_dogs.pth'   # backbone + classifier weights
LAC_CKPT     = 'dogs_lac_convnext_base_step010000.pth'
NC           = 120                                 # number of classes
N_SAMPLES    = 5
```

Output: `class_attribution_posneg.pdf`

Each row in the PDF corresponds to one sample:

```
[Original] [Stage 0 Inv.] … [Stage N-1 Inv.] [Class Recon.] [Positive α] [Negative α]
```

---

### 3. Per-Channel Spatial Basis Visualization (`channel_vis.py`)

Renders every channel's spatial basis `Ṽ_{l,c}` for a chosen stage and single input image.

```bash
python channel_vis.py
```

Configuration:

```python
MODEL      = 'convnext_base'
WEIGHTS    = 'weights/convnext_base_dogs.pth'
LAC_CKPT   = 'dogs_lac_convnext_base_step010000.pth'
STAGE_IDX  = -1       # stage to visualize; -1 = deepest
SAMPLE_IDX = 11       # index into the dataset
COLS, ROWS = 10, 10   # grid layout per PDF page
```

Output: `channel_vis_stage{STAGE_IDX}_{MODEL}.pdf`

- **Page 1:** Original image · full stage reconstruction · energy distribution bar chart (top-50 channels)
- **Pages 2+:** 10×10 grids of `Ṽ_{l,c}`, sorted by energy in descending order; each cell is labelled with channel index and energy value.

---

## Repository Structure

```
Supplementary Materials/
├── imagenet.py                              # Training script
├── stage_vis.py                             # Class attribution visualization
├── channel_vis.py                           # Per-channel basis visualization
├── imagenet_lac_convnext_base_step010000.pth  # Pretrained LAC weights (ConvNeXt-Base / ImageNet)
├── class_attribution_posneg.pdf             # Pre-generated class attribution results
└── channel_vis_stage3_convnext_base.pdf     # Pre-generated per-channel visualization
```

---

## Method Overview

```
Input image x
      │
      ▼
 FrozenEncoder (timm backbone, features_only)
      │  produces  h_stem,  h_1, h_2, …, h_L
      ▼
 Energy weighting  E_{l,c} = |h_{l,c}| / Σ_c |h_{l,c}|
      │
      ▼  for each stage l, top-k channels c:
 VJP chain:  h_l → [LAC_l] → h_{l-1} → … → [LAC_0] → h_stem → [LAC_stem] → pixel space
      │
      ▼
 Σ_c  E_{l,c} · Ṽ_{l,c}   =   X̂_l   (Stage-l reconstruction)
```

**LACRefiner** applies an instance-norm-equivalent GroupNorm with learnable `exp(log_γ)` scale and zero-mean `β` bias to each VJP, correcting the "Dirac spike" artefacts that arise when backpropagating through quantisation-style stem layers.

**Training** uses an L1 reconstruction loss on the *deepest* stage only (10 000 steps, batch size 20). All LAC modules receive gradients via the backpropagation chain through the deepest path.

**Inference** inverts all stages in a single forward pass without any per-image optimisation.

---

## Supported Backbones

Any `timm` CNN backbone with `features_only=True` support works out of the box:

| Family      | Example names |
|:------------|:--------------|
| ResNet      | `resnet18`, `resnet50`, `resnet101` |
| DenseNet    | `densenet121`, `densenet169` |
| ConvNeXt    | `convnext_base`, `convnext_large` |
| EfficientNet | `efficientnet_b0`, `tf_efficientnet_b4` |

To switch backbone, change `MODEL_NAME` (or `MODEL`) in the script configuration block. No other code changes are needed.

---



---

## License

This project is released under the [MIT License](LICENSE).
