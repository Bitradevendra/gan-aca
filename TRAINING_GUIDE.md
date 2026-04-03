# 🏋️ StyleGAN2-ADA-PyTorch — Complete Training Guide

> **Everything you need to train your own GAN from scratch or via transfer learning.**  
> **Project Path:** `C:\Users\iamir\Downloads\gan-aca`  
> **Last Updated:** 2026-02-23

---

## 📋 Table of Contents

1. [How Training Works (Theory)](#1-how-training-works-theory)
2. [Prerequisites & Hardware Requirements](#2-prerequisites--hardware-requirements)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Training Commands](#4-training-commands)
5. [Understanding Config Options](#5-understanding-config-options)
6. [ADA — Adaptive Augmentation Explained](#6-ada--adaptive-augmentation-explained)
7. [Transfer Learning (Recommended)](#7-transfer-learning-recommended)
8. [Monitoring Training Progress](#8-monitoring-training-progress)
9. [Checkpoints & Snapshots](#9-checkpoints--snapshots)
10. [Evaluating Quality (FID Score)](#10-evaluating-quality-fid-score)
11. [Using Your Trained Model](#11-using-your-trained-model)
12. [Training Tips & Best Practices](#12-training-tips--best-practices)
13. [Troubleshooting Training Issues](#13-troubleshooting-training-issues)
14. [Full Training Workflow (End-to-End)](#14-full-training-workflow-end-to-end)

---

## 1. How Training Works (Theory)

### 🎮 The Two-Player Game

StyleGAN2 training is a **minimax game** between two neural networks:

```
┌─────────────────────────────────────────────────────────────────┐
│  GENERATOR (G) — The Forger                                     │
│                                                                 │
│  Input:  Random noise z (512-dim Gaussian vector)               │
│  Process: z → Mapping Network → w → Synthesis Network → Image   │
│  Goal:   Create fake images so realistic that D can't tell      │
└───────────────────────────┬─────────────────────────────────────┘
                            │  fake images
                            ▼
              ┌─────────────────────────┐
              │    REAL images from     │
              │    your dataset     ────┤
              └─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  DISCRIMINATOR (D) — The Detective                              │
│                                                                 │
│  Input:  An image (real OR fake)                                │
│  Output: A scalar score (positive = real, negative = fake)      │
│  Goal:   Correctly identify real images and expose fakes        │
└─────────────────────────────────────────────────────────────────┘
```

### 🔄 The Training Loop (Every Step)

```
Each training iteration:

1. ─── D STEP ──────────────────────────────────────────────────
   a) Load batch of REAL images from dataset
   b) Generate batch of FAKE images via G(z)
   c) Apply ADA augmentation to BOTH real and fake
   d) Run both through D → get scores
   e) Compute D loss:
        L_D = softplus(-D(real)) + softplus(D(fake))
   f) Add R1 gradient penalty every 16 steps:
        L_R1 = γ/2 × ||∇_x D(real_aug)||²
   g) Backprop → Adam optimizer updates D weights

2. ─── G STEP ──────────────────────────────────────────────────
   a) Generate batch of FAKE images via G(z)
   b) Run through D → get scores
   c) Compute G loss:
        L_G = softplus(-D(G(z)))   ← G wants D to say "real"
   d) Add Path Length Regularization every 4 steps:
        L_PL = (||J^T y||₂ - a)²  ← smooth latent space
   e) Backprop → Adam optimizer updates G weights

3. ─── EMA UPDATE ──────────────────────────────────────────────
   G_ema ← lerp(G_ema, G, ema_beta)
   ↑ This smoothed model is what gets saved and used for inference

4. ─── ADA ADJUST ──────────────────────────────────────────────
   Monitor D's sign(D(real_aug)) → adjust augmentation probability p
   p++ if D overfitting, p-- if D underfitting
```

### 📐 Loss Functions Explained

| Loss | Formula | Purpose |
|------|---------|---------|
| **G Loss** | `softplus(-D(fake))` | G wants D to score fakes as real |
| **D Loss** | `softplus(-D(real)) + softplus(D(fake))` | D learns to separate real from fake |
| **R1 Penalty** | `γ/2 × ‖∇D(real)‖²` | Constrains D gradients → prevents mode collapse |
| **Path Length Reg** | `(‖J^T y‖ - a)²` | Makes W space smooth → better editability |

### 🧠 StyleGAN2 Generator Architecture

```
z (512-dim noise)
    │
    ▼
┌─────────────────────────────────────────────┐
│  MAPPING NETWORK (8 FC layers)              │
│  z → w  (both 512-dim)                      │
│  Learns a more disentangled space           │
└──────────────────┬──────────────────────────┘
                   │  w (repeated 18×)
                   ▼
┌─────────────────────────────────────────────┐
│  SYNTHESIS NETWORK (14 progressive blocks) │
│                                             │
│  4×4 → 8×8 → 16×16 → 32×32 → 64×64        │
│  → 128×128 → 256×256 → 512×512 → 1024×1024 │
│                                             │
│  Each block:                                │
│    ModulatedConv2d ← style from w           │
│    + Gaussian noise injection               │
│    + Leaky ReLU activation                  │
└─────────────────────────────────────────────┘
    │
    ▼
Output Image [3, H, W] in range [-1, 1]
```

---

## 2. Prerequisites & Hardware Requirements

### ✅ Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.7 – 3.9 | 3.8 recommended |
| CUDA | 11.1 – 11.8 | Must match PyTorch build |
| cuDNN | 8.x | Installed with CUDA toolkit |
| PyTorch | 1.7.1 – 1.12.1 | CUDA-specific build required |
| Visual C++ Build Tools | 2019+ | Windows only — for CUDA custom ops |

### 💻 GPU Requirements

| Use Case | Minimum VRAM | Recommended |
|----------|-------------|-------------|
| Training 256×256 | 4 GB | 8 GB |
| Training 512×512 | 8 GB | 16 GB |
| Training 1024×1024 | 16 GB | 24 GB+ |
| Inference only | 2 GB | 4 GB |

### ⏱️ Training Time Estimates

| Resolution | Images | GPU | Scratch | Transfer Learning |
|-----------|--------|-----|---------|-----------------|
| 256×256 | ~1,000 | RTX 3090 | ~8–16 hrs | **~1–2 hrs** ✅ |
| 256×256 | ~5,000 | RTX 3090 | ~24–48 hrs | **~4–8 hrs** ✅ |
| 512×512 | ~5,000 | RTX 3090 | ~3–5 days | ~12–24 hrs |
| 1024×1024 | ~70,000 | 8× A100 | ~1–2 weeks | ~2–3 days |

> 🚀 **Tip:** Always use Transfer Learning when possible — massively reduces training time.

---

## 3. Dataset Preparation

### 3.1 Image Requirements

Before training, your dataset **must** meet these requirements:

| Requirement | Specification |
|-------------|--------------|
| Shape | **Square only** (width = height) |
| Resolution | **Power of 2** — 64, 128, 256, 512, or 1024 |
| All images | **Same resolution** |
| Channels | RGB (3-channel) or Grayscale (1-channel) |
| Format | PNG, JPG, BMP, or other PIL-supported formats |
| Minimum count | ~1,000 (ADA handles small sets; 5,000+ recommended) |

### 3.2 Prepare and Convert Your Dataset

The `dataset_tool.py` script handles preparation for any image source:

#### From an Image Folder (Most Common)

```powershell
# Basic: folder → zip archive, auto-detect resolution
python dataset_tool.py --source=my_photos/ --dest=mydata.zip

# With center-crop to square + resize to 256px
python dataset_tool.py --source=my_photos/ --dest=mydata.zip \
    --transform=center-crop --width=256 --height=256

# Center-crop-wide (for wide/landscape images)
python dataset_tool.py --source=my_photos/ --dest=mydata.zip \
    --transform=center-crop-wide --width=512 --height=384

# Limit dataset size (e.g., use only first 2000 images)
python dataset_tool.py --source=my_photos/ --dest=mydata.zip \
    --max-images=2000 --transform=center-crop --width=256 --height=256
```

#### From Other Formats

```powershell
# From a zip archive
python dataset_tool.py --source=existing.zip --dest=mydata.zip \
    --transform=center-crop --width=256 --height=256

# From CIFAR-10
python dataset_tool.py --source=cifar-10-python.tar.gz --dest=cifar.zip

# From MNIST
python dataset_tool.py --source=train-images-idx3-ubyte.gz --dest=mnist.zip

# From LSUN (LMDB format)
python dataset_tool.py --source=cat_lmdb/ --dest=lsun_cat.zip \
    --transform=center-crop-wide --width=256 --height=256
```

### 3.3 Dataset Structure (for folder input)

```
my_photos/
├── img001.jpg
├── img002.jpg
├── img003.png
├── subdir/
│   ├── more_images.jpg    ← subdirectories are included recursively
│   └── ...
└── dataset.json           ← OPTIONAL: class labels for conditional training
```

**Optional `dataset.json` for conditional training:**
```json
{
    "labels": [
        ["img001.jpg", 0],
        ["img002.jpg", 1],
        ["img003.png", 0]
    ]
}
```

### 3.4 Verify Dataset

```powershell
# Quick check: does the dataset tool process it without errors?
python dataset_tool.py --source=my_photos/ --dest=check.zip \
    --max-images=10 --transform=center-crop --width=256 --height=256

# If successful: "Saving 10 images..." with no error = dataset is valid
```

---

## 4. Training Commands

### 4.1 Basic Training (1 GPU, from scratch)

```powershell
python train.py \
    --outdir=training-runs \
    --data=mydata.zip \
    --gpus=1 \
    --cfg=auto \
    --mirror=1
```

### 4.2 Training with ADA (recommended for small datasets)

```powershell
python train.py \
    --outdir=training-runs \
    --data=mydata.zip \
    --gpus=1 \
    --cfg=auto \
    --aug=ada \
    --mirror=1 \
    --snap=10
```

### 4.3 Multi-GPU Training

```powershell
# 2 GPUs
python train.py --outdir=training-runs --data=mydata.zip --gpus=2 --cfg=auto

# 4 GPUs
python train.py --outdir=training-runs --data=mydata.zip --gpus=4 --cfg=auto

# NOTE: --gpus must be a power of 2 (1, 2, 4, 8)
```

### 4.4 Conditional Training (with class labels)

```powershell
python train.py \
    --outdir=training-runs \
    --data=mydata.zip \
    --gpus=1 \
    --cfg=auto \
    --cond=1 \       ← requires dataset.json with labels
    --mirror=1
```

### 4.5 Resume an Interrupted Run

```powershell
# Resume from last checkpoint automatically
python train.py \
    --outdir=training-runs \
    --data=mydata.zip \
    --gpus=1 \
    --cfg=auto \
    --resume=training-runs/00000-mydata-auto1/network-snapshot-000100.pkl
```

### 4.6 Dry Run (test config without training)

```powershell
# Validates all settings and prints full config, then exits
python train.py \
    --outdir=training-runs \
    --data=mydata.zip \
    --gpus=1 \
    --cfg=auto \
    --dry-run
```

---

## 5. Understanding Config Options

### 5.1 Base Configurations (`--cfg`)

| Config | Best For | Batch | Features |
|--------|---------|-------|---------|
| `auto` | **Everything (default)** — auto-tunes based on resolution + GPU count | dynamic | Recommended starting point |
| `stylegan2` | Reproducing paper results at 1024×1024, 8 GPUs | 32 | Standard StyleGAN2 Config F |
| `paper256` | FFHQ/LSUN at 256×256, 8 GPUs | 64 | Paper-accurate 256px |
| `paper512` | AFHQ/BreCaHAD at 512×512, 8 GPUs | 64 | Paper-accurate 512px |
| `paper1024` | MetFaces at 1024×1024, 8 GPUs | 32 | Paper-accurate 1024px |
| `cifar` | CIFAR-10 at 32×32, 2 GPUs | 64 | No path length reg, no style mixing |

### 5.2 All Training Parameters

```
GENERAL:
  --outdir         Where to save training run directories        [REQUIRED]
  --gpus           Number of GPUs (must be power of 2)          [default: 1]
  --snap           Save snapshot every N ticks                   [default: 50]
  --metrics        Metrics to evaluate: fid50k_full, kid50k_full [default: fid50k_full]
  --seed           Random seed                                   [default: 0]

DATASET:
  --data           Path to dataset (folder or .zip)             [REQUIRED]
  --cond           Enable conditional training (needs labels)    [default: False]
  --subset         Train on only N images                        [default: all]
  --mirror         Horizontal flip augmentation                  [default: False]

BASE CONFIG:
  --cfg            Base configuration preset                     [default: auto]
  --gamma          Override R1 gamma (regularization strength)   [auto]
  --kimg           Override total training duration (kimgs)      [default: 25000]
  --batch          Override global batch size                    [auto]

AUGMENTATION:
  --aug            Augmentation mode: ada / noaug / fixed        [default: ada]
  --p              Augmentation probability (only for --aug=fixed)
  --target         ADA target r_t value                          [default: 0.6]
  --augpipe        Augmentation pipeline                         [default: bgc]

TRANSFER LEARNING:
  --resume         Resume from checkpoint or pretrained model
  --freezed        Number of D layers to freeze (Freeze-D)       [default: 0]

PERFORMANCE:
  --fp32           Disable FP16 mixed precision                  [default: False]
  --nhwc           Use NHWC layout with FP16                     [default: False]
  --nobench        Disable cuDNN benchmarking                    [default: False]
  --allow-tf32     Allow TF32 for matmul/convolutions            [default: False]
  --workers        Number of DataLoader workers                  [default: 3]
```

### 5.3 Augmentation Pipelines (`--augpipe`)

| Pipeline | Operations | Description |
|----------|-----------|-------------|
| `blit` | xflip, rotate90, xint | Blitting operations |
| `geom` | scale, rotate, aniso, xfrac | Geometric transforms |
| `color` | brightness, contrast, lumaflip, hue, saturation | Color transforms |
| `filter` | imgfilter | Image filtering |
| `noise` | noise | Additive noise |
| `cutout` | cutout | Random cutout |
| `bg` | blit + geom | Combined |
| `bgc` ✅ | blit + geom + color | **Default — balanced** |
| `bgcf` | bgc + filter | More aggressive |
| `bgcfn` | bgcf + noise | Even more aggressive |
| `bgcfnc` | bgcfn + cutout | Maximum augmentation |

---

## 6. ADA — Adaptive Augmentation Explained

### What Problem Does ADA Solve?

```
WITHOUT ADA (small dataset, ~1000 images):
  Epoch 1:  D learns some real features
  Epoch 5:  D starts memorizing exact training images
  Epoch 10: D completely memorizes dataset → overfits
  Result:   G can't learn → training collapses

WITH ADA:
  Augmentation makes real images "harder" to memorize
  D must learn genuine features, not memorized pixels
  G can continuously improve
  Works with as few as ~1,000 images!
```

### How ADA Adapts

```
Every 4 training steps, ADA monitors:
  r_t = E[sign(D(augmented real images))]

  r_t ≈ 1.0  → D always says "real" → D is overfitting
  r_t ≈ 0.0  → D is random → D is underfitting
  r_t ≈ 0.6  → Healthy balance (default target)

Adjustment rule:
  p += sign(r_t - 0.6) × (batch_size / 500,000)

p is clamped to [0.0, 1.0]
```

### ADA in Practice

```
Early training (low p):
  Augmentation probability ≈ 0–30%
  D learns basic real vs fake features

Mid training (adapted p):
  If dataset is small: p might rise to 60–80%
  If dataset is large: p stays low (5–20%)
  ADA automatically finds the right level

Late training:
  p stabilizes once G produces realistic enough images
```

### ADA Modes

```powershell
# Recommended: fully adaptive (automatic)
--aug=ada

# No augmentation (only for very large datasets, 100k+ images)
--aug=noaug

# Fixed augmentation probability (manual control)
--aug=fixed --p=0.5       ← 50% augmentation probability always

# Custom ADA target (default 0.6)
--aug=ada --target=0.7    ← allow D to be slightly more confident
```

---

## 7. Transfer Learning (Recommended)

Transfer learning **starts G and D from pretrained weights** instead of random initialization.

### 🚀 Why Use Transfer Learning?

| Metric | From Scratch | Transfer Learning |
|--------|-------------|------------------|
| Training time | Days–weeks | Hours |
| Min images needed | ~5,000+ | **~100–500** |
| Quality at 1k kimgs | Poor | **Already good** |
| Risk of failure | High | Low |

### 7.1 Built-in Pretrained Sources

```powershell
# Face images at 256×256 (from FFHQ dataset)
--resume=ffhq256

# Face images at 512×512
--resume=ffhq512

# Face images at 1024×1024
--resume=ffhq1024

# CelebA-HQ faces at 256×256
--resume=celebahq256

# Dog images at 256×256
--resume=lsundog256
```

### 7.2 Transfer From Your Own Checkpoint

```powershell
# Resume from a specific .pkl checkpoint
--resume=training-runs/00000-.../network-snapshot-000100.pkl

# Resume from any URL
--resume=https://example.com/my_model.pkl
```

### 7.3 Transfer Learning Commands

```powershell
# ── Most Common: Transfer from FFHQ faces to your face dataset ──
python train.py \
    --outdir=training-runs \
    --data=my_faces.zip \
    --gpus=1 \
    --resume=ffhq256 \
    --cfg=paper256 \
    --snap=10 \          ← save more often — quality improves fast
    --mirror=1

# ── Transfer to animal/object images (less related domain) ──
python train.py \
    --outdir=training-runs \
    --data=my_cats.zip \
    --gpus=1 \
    --resume=ffhq256 \
    --cfg=auto \
    --kimg=5000 \         ← shorter training usually enough

# ── Freeze-D: Freeze D layers to preserve pretrained features ──
# Useful when your dataset is very small (<500 images)
python train.py \
    --outdir=training-runs \
    --data=my_tiny_dataset.zip \
    --gpus=1 \
    --resume=ffhq256 \
    --cfg=paper256 \
    --freezed=10 \        ← freeze 10 D layers
    --snap=5
```

### 7.4 Transfer Learning Tips

```
✅ DO:
  - Use the pretrained model closest to your target domain
  - Save snapshots frequently (--snap=5 or --snap=10)
  - Stop training once quality peaks (check FID score)
  - Use --mirror=1 for face datasets (x-flip symmetry)

❌ DON'T:
  - Use ffhq1024 as resume for a 256×256 dataset (resolution mismatch)
  - Train too long — transfer models can overfit faster
  - Use --aug=noaug with small datasets even in transfer mode
```

---

## 8. Monitoring Training Progress

### 8.1 Reading the Training Log

The training log (`log.txt`) entries look like this:

```
tick 1    kimg 0.1    lod 0.00    minibatch 8    time 0m 12s    sec/tick 12.1    sec/kimg 121.4    maintenance 0.6    cpumem 4.62    gpumem 7.23    augment 0.000
tick 10   kimg 1.0    lod 0.00    minibatch 8    time 2m 07s    sec/tick 12.2    sec/kimg 122.1    maintenance 0.5    cpumem 4.65    gpumem 7.25    augment 0.022
tick 50   kimg 5.0    ...         FID: 245.32
tick 100  kimg 10.0   ...         FID: 158.12
tick 200  kimg 20.0   ...         FID: 87.45
tick 500  kimg 50.0   ...         FID: 42.18     ← getting good
tick 1000 kimg 100.0  ...         FID: 18.75     ← very good
```

**Key Metrics to Watch:**

| Metric | What it means | Healthy range |
|--------|--------------|---------------|
| `augment` | Current ADA probability p | 0.0–0.8 (auto-adapts) |
| `gpumem` | GPU memory usage (GB) | Below your GPU's VRAM |
| `sec/kimg` | Seconds per 1000 images processed | Depends on GPU |
| `FID` | Image quality score | **Lower is better** (< 30 = great) |

### 8.2 Signs of Healthy Training

```
✅ HEALTHY:
  - FID decreasing over time (even slowly)
  - augment value between 0.0 and 0.8
  - gpumem stable, no spikes
  - Generated images getting more detailed over snapshots

⚠️ WARNING:
  - FID stops improving for 100+ ticks → stop early, use best snapshot
  - augment suddenly jumps to 1.0 → D is severely overfitting
  - FID gets worse after a certain point → you passed the best checkpoint

❌ FAILURE:
  - Loss becomes NaN → numerical instability (try --fp32)
  - All generated images look identical → mode collapse
  - FID > 300 after 100+ kimgs → training is not working
```

### 8.3 Check Progress Visually

```powershell
# Generate sample images from a snapshot mid-training
python generate.py \
    --network=training-runs/00000-.../network-snapshot-000050.pkl \
    --seeds=1-16 \
    --trunc=0.7 \
    --outdir=progress_check

# Open the out/ folder and visually inspect quality
```

### 8.4 Monitor with Python Script

```powershell
# Quick one-liner to watch the last line of log.txt
# (Run in a separate PowerShell window)
Get-Content training-runs\00000-mydata-auto1\log.txt -Wait -Tail 5
```

---

## 9. Checkpoints & Snapshots

### 9.1 Snapshot Contents

```
training-runs/
└── 00000-mydata-auto1/
    ├── network-snapshot-000000.pkl    ← tick 0 (initial state)
    ├── network-snapshot-000050.pkl    ← every --snap ticks
    ├── network-snapshot-000100.pkl
    ├── ...
    ├── log.txt                        ← metric history per tick
    ├── stats.jsonl                    ← machine-readable metrics
    └── training_options.json          ← exact config used (reproducibility)
```

Each `.pkl` snapshot contains:
```python
{
  'G':       G,        # Current generator weights
  'D':       D,        # Current discriminator weights
  'G_ema':   G_ema,    # ← EMA-smoothed G — USE THIS for inference!
  'G_opt':   G_opt,    # Adam optimizer state for G
  'D_opt':   D_opt,    # Adam optimizer state for D
  'augment_pipe': ..., # Current ADA state
  'training_set_kwargs': ...,
  'cur_nimg': ...,     # How many images have been processed
  'cur_tick': ...,     # Current tick number
}
```

> 💡 **`G_ema` is the model used for generation.** It's a moving average of G weights — smoother and more stable than raw G.

### 9.2 Choosing the Best Snapshot

```
Strategy 1 — FID-based (most reliable):
  → Read log.txt, find the tick where FID was lowest
  → Use that snapshot: network-snapshot-{tick:06d}.pkl

Strategy 2 — Visual inspection:
  → Generate 16 images from each snapshot (every 50 ticks)
  → Pick the one with best visual quality

Strategy 3 — Conservative:
  → For transfer learning: stop around 500–2000 kimgs
  → For scratch: stop around 5000–25000 kimgs
  → Very long training often leads to overfitting
```

### 9.3 Find Best FID from Log

```powershell
# Search log.txt for FID scores
Select-String -Path "training-runs\00000-mydata-auto1\log.txt" -Pattern "FID"
```

---

## 10. Evaluating Quality (FID Score)

### 10.1 What is FID?

**Fréchet Inception Distance (FID)** measures how close the distribution of generated images is to real images.

```
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))

Lower FID = better quality
```

| FID Range | Quality Level |
|-----------|--------------|
| < 5 | Exceptional (near real) |
| 5 – 15 | Excellent |
| 15 – 30 | Very good |
| 30 – 50 | Good |
| 50 – 100 | Acceptable |
| > 100 | Poor / early training |
| > 200 | Very poor / training issues |

### 10.2 Calculate FID Manually

```powershell
# Calculate FID for a specific snapshot
python calc_metrics.py \
    --metrics=fid50k_full \
    --network=training-runs/00000-.../network-snapshot-000100.pkl \
    --data=mydata.zip \
    --gpus=1

# Calculate multiple metrics at once
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=training-runs/00000-.../network-snapshot-000100.pkl \
    --data=mydata.zip \
    --gpus=1
```

### 10.3 Available Metrics

| Metric | Description |
|--------|-------------|
| `fid50k_full` | FID over 50k samples vs full dataset |
| `kid50k_full` | Kernel Inception Distance (unbiased FID variant) |
| `pr50k3_full` | Precision & Recall |
| `is50k` | Inception Score |
| `ppl_zfull` | Path Length Perceptual distance (Z space) |
| `ppl_wfull` | Path Length Perceptual distance (W space) |

---

## 11. Using Your Trained Model

### 11.1 Generate Images from Your Model

```powershell
# Generate 20 random images
python generate.py \
    --network=training-runs/00000-mydata-auto1/network-snapshot-000100.pkl \
    --seeds=1-20 \
    --trunc=0.7 \
    --outdir=my_results

# Generate with full diversity (no truncation)
python generate.py \
    --network=training-runs/.../network-snapshot-000100.pkl \
    --seeds=0-99 \
    --trunc=1.0 \
    --outdir=my_results
```

### 11.2 Style Mixing with Your Model

```powershell
python style_mixing.py \
    --network=training-runs/.../network-snapshot-000100.pkl \
    --rows=1,2,3,4 \
    --cols=10,20,30 \
    --styles=0-6 \
    --outdir=style_mix_results
```

### 11.3 Explore Latent Space of Your Model

```powershell
# Discover editing directions in your model
python explore_latent.py discover \
    --network=training-runs/.../network-snapshot-000100.pkl \
    --outdir=latent_explore \
    --seeds=50,100,150 \
    --num-directions=10

# Edit along discovered directions
python explore_latent.py edit \
    --network=training-runs/.../network-snapshot-000100.pkl \
    --outdir=latent_explore \
    --seed=100 \
    --direction=0 \
    --strength="-5,5"
```

### 11.4 Project an Image into Your Model's Latent Space

```powershell
python projector.py \
    --network=training-runs/.../network-snapshot-000100.pkl \
    --target=my_image.png \
    --outdir=out \
    --num-steps=1000
```

### 11.5 Serve Your Model via Web UI

```powershell
# Edit server.py line 24 to point to your model:
# NETWORK_PKL = os.path.join(BASE_DIR, "training-runs/.../network-snapshot-000100.pkl")

python server.py
# Open: http://localhost:5000
```

---

## 12. Training Tips & Best Practices

### ✅ General Best Practices

```
1. ALWAYS start with transfer learning if your domain is face-like
   → Use --resume=ffhq256 for 256px faces

2. Use --mirror=1 for face datasets
   → Doubles effective dataset size for free

3. Save snapshots frequently during transfer learning
   → --snap=5 or --snap=10 (quality peaks early)

4. Monitor FID score — don't rely on visual inspection alone
   → FID can reveal issues invisible to the eye

5. Use ZIP datasets, not raw folders, for performance
   → Zip archives are faster to load during training

6. Keep backup of best snapshot
   → Copy best network-snapshot-XXXXXX.pkl somewhere safe
```

### 🎯 Hyperparameter Tuning

| Scenario | Recommendation |
|---------|---------------|
| < 500 images | `--resume=ffhq256 --freezed=10 --snap=5` |
| 500–2,000 images | `--resume=ffhq256 --aug=ada --snap=10` |
| 2,000–10,000 images | `--cfg=auto --aug=ada --mirror=1` |
| > 10,000 images | `--cfg=auto --aug=noaug` or `--aug=ada` |
| Different domain than faces | `--resume=lsundog256` or no resume |
| Conditional model | `--cond=1` (requires labels in dataset.json) |

### 🔧 Performance Optimization

```powershell
# Increase workers for faster data loading
python train.py ... --workers=4

# Use NHWC layout on A100/RTX30xx for speed (experimental)
python train.py ... --nhwc=True

# Allow TF32 on Ampere GPUs (small accuracy trade-off for speed)
python train.py ... --allow-tf32=True

# Reduce batch size if running out of VRAM
python train.py ... --batch=8
```

---

## 13. Troubleshooting Training Issues

### ❌ CUDA Out of Memory (OOM)

```
Error: CUDA out of memory. Tried to allocate X GB

Fixes:
1. Reduce batch size:        --batch=8  (or --batch=4)
2. Reduce resolution:        Use 256×256 dataset instead of 512×512
3. Disable NHWC:             Remove --nhwc flag
4. Use FP32 mode:            --fp32=True  (uses more memory but avoids some ops)
5. Reduce workers:           --workers=1
```

### ❌ Custom CUDA Ops Fail to Compile

```
Error: Failed to build custom op...

Windows Fixes:
1. Install Visual C++ Build Tools 2019+:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Set environment variable:
   $env:DISTUTILS_USE_SDK = 1

3. Run from Visual Studio Developer Command Prompt

4. Or use --fp32 to avoid CUDA custom ops:
   python train.py ... --fp32=True
```

### ❌ Training Loss is NaN

```
Error: Loss is NaN

Fixes:
1. Use FP32 mode (most common fix):
   --fp32=True

2. Lower learning rate with smaller gamma:
   --gamma=0.5  (or try 0.1)

3. Reduce batch size:
   --batch=8

4. Use a smaller/simpler dataset to test
```

### ❌ Mode Collapse (All Images Look the Same)

```
Symptom: Generated images are nearly identical despite different seeds

Fixes:
1. Increase R1 gamma:        --gamma=20 (stronger D regularization)
2. Enable ADA if not already: --aug=ada
3. Restart from earlier snapshot + use fresh init
4. Reduce --kimg (may have overtrained)
5. Check dataset diversity — may need more varied images
```

### ❌ D Always Winning (G Never Improves)

```
Symptom: FID stays > 200 for many ticks, generated images are noise

Fixes:
1. Check ADA is enabled:     --aug=ada
2. Increase augmentation:    --aug=fixed --p=0.5
3. Decrease D strength:      --gamma=0.01
4. Use transfer learning:    --resume=ffhq256
5. Increase dataset diversity (D may be too confident with few images)
```

### ❌ Very Slow Training

```
Symptom: sec/kimg is very high (> 300)

Fixes:
1. Increase DataLoader workers:  --workers=4
2. Use ZIP dataset instead of folder
3. Enable cuDNN benchmarking (default is ON, don't use --nobench)
4. Enable TF32:                  --allow-tf32=True
5. Check GPU utilization:        nvidia-smi
6. Ensure data is on SSD, not HDD
```

### ❌ Resolution Mismatch with Resume

```
Error: Network resolution doesn't match dataset resolution

Fix: Use the matching pretrained model:
  256×256 dataset → --resume=ffhq256
  512×512 dataset → --resume=ffhq512
  1024×1024 dataset → --resume=ffhq1024
  
Or don't use --resume and train from scratch.
```

---

## 14. Full Training Workflow (End-to-End)

Here is the **complete workflow** from raw photos to a fully usable model:

```powershell
# ═══════════════════════════════════════════════════════════
# STEP 1 — Navigate to project directory
# ═══════════════════════════════════════════════════════════
cd C:\Users\iamir\Downloads\gan-aca

# Activate virtual environment
.venv\Scripts\activate

# ═══════════════════════════════════════════════════════════
# STEP 2 — Prepare your dataset
# ═══════════════════════════════════════════════════════════
python dataset_tool.py ^
    --source=my_photos/ ^
    --dest=mydata.zip ^
    --transform=center-crop ^
    --width=256 ^
    --height=256

# ═══════════════════════════════════════════════════════════
# STEP 3 — Dry run to validate config
# ═══════════════════════════════════════════════════════════
python train.py ^
    --outdir=training-runs ^
    --data=mydata.zip ^
    --gpus=1 ^
    --cfg=auto ^
    --resume=ffhq256 ^
    --mirror=1 ^
    --snap=10 ^
    --dry-run

# Review the printed config. If it looks correct, proceed.

# ═══════════════════════════════════════════════════════════
# STEP 4 — Start training (Transfer Learning recommended)
# ═══════════════════════════════════════════════════════════
python train.py ^
    --outdir=training-runs ^
    --data=mydata.zip ^
    --gpus=1 ^
    --cfg=auto ^
    --resume=ffhq256 ^
    --aug=ada ^
    --mirror=1 ^
    --snap=10

# Training will start. You'll see:
#   tick 1   kimg 0.1   augment 0.000   ...
#   tick 10  kimg 1.0   augment 0.015   ...
#   tick 50  kimg 5.0   FID: 145.3      ...

# ═══════════════════════════════════════════════════════════
# STEP 5 — Monitor (in a separate terminal)
# ═══════════════════════════════════════════════════════════
Get-Content training-runs\00000-mydata-auto1\log.txt -Wait -Tail 3

# ═══════════════════════════════════════════════════════════
# STEP 6 — Visually check intermediate results
# ═══════════════════════════════════════════════════════════
python generate.py ^
    --network=training-runs\00000-mydata-auto1\network-snapshot-000050.pkl ^
    --seeds=1-16 ^
    --trunc=0.7 ^
    --outdir=progress_check_050

python generate.py ^
    --network=training-runs\00000-mydata-auto1\network-snapshot-000100.pkl ^
    --seeds=1-16 ^
    --trunc=0.7 ^
    --outdir=progress_check_100

# Open the folders — pick the snapshot where images look best

# ═══════════════════════════════════════════════════════════
# STEP 7 — Stop training at the best checkpoint
# ═══════════════════════════════════════════════════════════
# Press Ctrl+C when satisfied with quality OR when FID stops improving
# Copy the best snapshot to safety:
Copy-Item training-runs\00000-mydata-auto1\network-snapshot-000100.pkl best_model.pkl

# ═══════════════════════════════════════════════════════════
# STEP 8 — Use your trained model
# ═══════════════════════════════════════════════════════════

# Generate images
python generate.py --network=best_model.pkl --seeds=1-50 --trunc=0.7 --outdir=final_output

# Explore latent space
python explore_latent.py discover --network=best_model.pkl --outdir=latent_explore --seeds=10,20,30

# Style mixing
python style_mixing.py --network=best_model.pkl --rows=1,2,3 --cols=10,20,30 --outdir=style_mix_results

# Project a custom image
python projector.py --network=best_model.pkl --target=my_photo.png --outdir=out

# Run Web UI
python server.py   # Open http://localhost:5000
```

---

## 📊 Quick Decision Chart

```
Do you have a face dataset?
  ├── YES → Use --resume=ffhq256 (or ffhq512/ffhq1024 for higher res)
  └── NO  → Use --resume=lsundog256 or train from scratch

How many images?
  ├── < 500   → Transfer learning REQUIRED + --freezed=10
  ├── 500–5k  → Transfer learning strongly recommended + --aug=ada
  ├── 5k–50k  → Transfer learning optional + --aug=ada
  └── > 50k   → Train from scratch optional + --aug=noaug

What resolution?
  ├── 256×256  → --cfg=paper256 or --cfg=auto + --resume=ffhq256
  ├── 512×512  → --cfg=paper512 or --cfg=auto + --resume=ffhq512
  └── 1024×1024 → --cfg=paper1024 or --cfg=auto + --resume=ffhq1024
```

---

*Training Guide for StyleGAN2-ADA-PyTorch — `C:\Users\iamir\Downloads\gan-aca`*  
*Reference: NVlabs/stylegan2-ada-pytorch paper — "Training Generative Adversarial Networks with Limited Data"*
