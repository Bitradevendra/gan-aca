# 🎨 StyleGAN2-ADA-PyTorch — Full Project Documentation

> **Project:** Generative Adversarial Network — Adaptive Discriminator Augmentation  
> **Base Repo:** [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)  
> **Local Path:** `C:\Users\iamir\Downloads\gan-aca`  
> **Pre-trained Model:** `network.pkl` ✅ (already downloaded, ~282 MB)  
> **Last Updated:** 2026-02-23

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Environment Setup (pip-to-pin)](#4-environment-setup-pip-to-pin)
5. [Pipeline: End-to-End Workflow](#5-pipeline-end-to-end-workflow)
6. [Module Reference](#6-module-reference)
   - [generate.py](#generatepy)
   - [style_mixing.py](#style_mixingpy)
   - [explore_latent.py](#explore_latentpy)
   - [projector.py](#projectorpy)
   - [dataset_tool.py](#dataset_toolpy)
   - [train.py](#trainpy)
   - [server.py](#serverpy)
7. [Web UI Backend](#7-web-ui-backend)
8. [Output Folders Reference](#8-output-folders-reference)
9. [Key Concepts Explained](#9-key-concepts-explained)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

StyleGAN2-ADA-PyTorch is NVIDIA's state-of-the-art Generative Adversarial Network framework. It implements:

- **StyleGAN2** — A style-based generator architecture that produces high-resolution, photorealistic images
- **ADA (Adaptive Discriminator Augmentation)** — Automatically adapts the augmentation probability during training, enabling training on **limited data** (as few as ~1,000 images)
- **Custom Web UI** — A Flask-based backend + HTML/JS frontend for interactive exploration

### 🔑 What This Project Can Do

| Feature | Script | Description |
|---------|--------|-------------|
| Random Image Generation | `generate.py` | Generate any number of unique faces/images from seeds |
| Style Mixing | `style_mixing.py` | Combine coarse features from one image with fine features of another |
| Latent Space Discovery | `explore_latent.py discover` | Find meaningful editing directions (age, pose, smile, etc.) using SeFa |
| Latent Space Editing | `explore_latent.py edit` | Apply discovered directions to manipulate specific attributes |
| Face Morphing | `explore_latent.py interpolate` | Create smooth animations between two generated faces |
| Image Projection | `projector.py` | Encode YOUR OWN photo into the GAN's latent space |
| Custom Training | `train.py` | Train a new GAN from scratch on your own dataset |
| Dataset Preparation | `dataset_tool.py` | Convert image folders/archives into training-ready format |
| Web Interface | `server.py` | Run a local web server to do all of the above via a browser UI |

---

## 2. Project Structure

```
gan-aca/
│
├── 📄 generate.py          # Random image generation from seeds
├── 📄 style_mixing.py      # Style mixing matrix generation
├── 📄 explore_latent.py    # Latent space exploration (SeFa): discover / edit / interpolate
├── 📄 projector.py         # Project a real image into W latent space
├── 📄 dataset_tool.py      # Dataset preparation and conversion tool
├── 📄 train.py             # Full GAN training loop
├── 📄 server.py            # Flask web server (API + UI backend)
├── 📄 legacy.py            # Backwards-compatible network loader
├── 📄 calc_metrics.py      # FID/KID/precision/recall evaluation
│
├── 📁 dnnlib/              # NVIDIA deep learning utilities
│   ├── util.py             # URL opener, class constructor helpers
│   └── ...
│
├── 📁 torch_utils/         # Custom PyTorch ops (CUDA kernels, persistence, etc.)
│   ├── custom_ops.py
│   ├── persistence.py
│   ├── training_stats.py
│   └── ops/                # Upfirdn2d, bias_act CUDA kernels
│
├── 📁 training/            # Training infrastructure
│   ├── networks.py         # Generator (G) and Discriminator (D) architectures
│   ├── training_loop.py    # Main training loop
│   ├── loss.py             # StyleGAN2Loss (non-saturating + R1 regularization)
│   ├── augment.py          # ADA augmentation pipeline
│   └── dataset.py          # Dataset loader (ImageFolderDataset)
│
├── 📁 metrics/             # Evaluation metrics
│   ├── metric_main.py      # Orchestrator
│   ├── frechet_inception_distance.py
│   ├── kernel_inception_distance.py
│   ├── precision_recall.py
│   └── ...
│
├── 📁 ui/                  # Web frontend
│   ├── index.html          # Single-page app
│   ├── style.css           # UI styling
│   └── app.js              # Frontend JavaScript logic
│
├── 📁 out/                 # Generated images output folder
├── 📁 style_mix_results/   # Style mixing output folder
├── 📁 latent_explore/      # Latent exploration output folder
│
├── 📄 network.pkl          # Pre-trained model weights (~282 MB)
├── 📄 requirements.txt     # Python dependencies
├── 📄 COMMANDS.md          # Quick command reference
├── 📄 PROJECT_DOCUMENTATION.md  # This file
└── 📄 Dockerfile           # Docker container definition
```

---

## 3. Architecture Deep Dive

### 3.1 The Two-Network GAN Structure

```
┌─────────────────────────────────────────────────────────┐
│                    GENERATOR (G)                         │
│                                                          │
│   z (noise) ──► [Mapping Network] ──► w (style code)    │
│                       ↓                                  │
│   w ──► [Synthesis Network] ──► Generated Image          │
│          (18 progressive layers: 4x4 → 1024x1024)        │
└─────────────────────────────────────────────────────────┘
                         │
                    (fake image)
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  DISCRIMINATOR (D)                        │
│   Real or Fake image ──► CNN ──► score (real/fake)       │
│   ADA augments both real & fake images to fight overfit  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Latent Spaces

| Space | Symbol | Dimensions | Description |
|-------|--------|-----------|-------------|
| Noise space | **Z** | 512-dim | Random Gaussian input. NOT meaningful directly |
| Intermediate space | **W** | 512-dim | After mapping network. More disentangled, better for editing |
| Extended space | **W+** | 18 × 512 | One W vector per synthesis layer. Best for projection |

### 3.3 SeFa — Closed-Form Factorization (used in `explore_latent.py`)

SeFa finds meaningful editing directions **without any labels or training**:

```
1. Extract weight matrix A from synthesis network's affine/modulation layers
2. Compute  W^T @ W  (covariance-like matrix)
3. Run  eigendecomposition  →  eigenvectors = editing directions
4. Eigenvector with largest eigenvalue = most impactful direction
5. Add  α × direction  to W latent code  →  controlled attribute change
```

### 3.4 Image Generation Pipeline

```
Seed (int)
    │
    ▼
np.random.RandomState(seed).randn(1, 512)   ← deterministic random noise z
    │
    ▼
G.mapping(z, label)   ← maps z → w (18 × 512)
    │
    ▼
w_trunc = w_avg + (w - w_avg) * truncation_psi   ← optional truncation
    │
    ▼
G.synthesis(w, noise_mode)   ← generates image tensor [1, 3, H, W]
    │
    ▼
(img * 127.5 + 128).clamp(0, 255)   ← denormalize to [0, 255]
    │
    ▼
PIL.Image.fromarray(...)   ← save as PNG
```

---

## 4. Environment Setup (pip-to-pin)

### 4.1 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.7 | 3.8 – 3.9 |
| CUDA | 11.1 | 11.3 |
| GPU VRAM | 4 GB | 8+ GB |
| GPU | NVIDIA only | RTX 2080+ |
| OS | Windows / Linux | Ubuntu 20.04 |

### 4.2 Step-by-Step Installation

#### Step 1 — Create a Virtual Environment

```powershell
# Navigate to the project
cd C:\Users\iamir\Downloads\gan-aca

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate
```

#### Step 2 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

#### Step 3 — Install PyTorch (CUDA 11.3 example)

```powershell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

> ⚠️ **Critical:** PyTorch version must match your CUDA version exactly.  
> Check your CUDA: `nvcc --version`  
> Find compatible versions: https://pytorch.org/get-started/previous-versions/

#### Step 4 — Install All Other Dependencies

```powershell
pip install -r requirements.txt
```

#### Step 5 — Verify Installation

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: True
```

#### Step 6 — Test the Model

```powershell
python generate.py --network=network.pkl --seeds=1,2,3 --outdir=out
```

### 4.3 Pinned Dependency Logic (pip-to-pin Explained)

The term **pip-to-pin** refers to the practice of:

```
pip install <package>        ← unpinned (gets latest, risky)
     ↓ test & verify
pip freeze > requirements.txt  ← pin exact working versions
     ↓ share
pip install -r requirements.txt  ← reproducible install for everyone
```

This project's `requirements.txt` is **pinned** — every version is locked to ensure the exact same environment is reproduced on any machine.

---

## 5. Pipeline: End-to-End Workflow

### 🔵 Pipeline A — Generate Random Art

```
[Start]
   │
   ▼
python generate.py
   --network=network.pkl
   --seeds=1-100          ← pick any seed integers
   --trunc=0.7            ← 0.5–1.0 range (lower = safer faces)
   --outdir=out
   │
   ▼
[out/seed0001.png, seed0002.png, ...]
```

### 🟣 Pipeline B — Style Mixing

```
[Choose row seeds]  +  [Choose col seeds]
        │                       │
        └───────────┬───────────┘
                    ▼
python style_mixing.py
   --rows=85,100,75    ← provide structure/pose
   --cols=55,821,293   ← provide color/texture
   --styles=0-6        ← which layers to mix
   --outdir=style_mix_results
                    │
                    ▼
[style_mix_results/grid.png]  ← full mixing matrix
[style_mix_results/85-55.png] ← individual mixed images
```

### 🟠 Pipeline C — Latent Space Exploration (SeFa)

```
STEP 1 — Discover Directions:
python explore_latent.py discover
   --network=network.pkl
   --outdir=latent_explore
   --seeds=100,200,300
        │
        ▼
[latent_explore/discover_seed100.png]  ← direction grid
[latent_explore/sefa_directions.npz]   ← saved for reuse

STEP 2 — Edit Along a Direction:
python explore_latent.py edit
   --seed=200
   --direction=0          ← most impactful direction
   --strength="-6,6"
   --layers=0-7
        │
        ▼
[latent_explore/edit_strip_seed200_dir0.png]

STEP 3 — Interpolate / Morph:
python explore_latent.py interpolate
   --seed-start=100
   --seed-end=300
   --frames=60
   --mode=slerp
   --loop
        │
        ▼
[latent_explore/morph_100_to_300/frame_0000.png ... frame_0059.png]
[latent_explore/morph_100_to_300/filmstrip_preview.png]

STEP 4 — Convert to Video (FFmpeg):
ffmpeg -framerate 30 -i latent_explore/morph_100_to_300/frame_%04d.png
       -c:v libx264 -pix_fmt yuv420p latent_explore/morph.mp4
```

### 🟢 Pipeline D — Project Your Own Photo

```
[YOUR PHOTO: my_face.png]
        │
        ▼
python projector.py
   --network=network.pkl
   --target=my_face.png
   --num-steps=1000     ← more steps = better accuracy (slower)
   --outdir=out
        │
        ▼
[out/projected_w.npz]   ← your face as a latent code
[out/proj.png]          ← reconstructed face
[out/proj.mp4]          ← optimization progress video
[out/target.png]        ← your original (resized)

        │
        ▼  (optional: regenerate from latent code)
python generate.py
   --network=network.pkl
   --projected-w=out/projected_w.npz
   --outdir=out
```

### 🔴 Pipeline E — Train on Your Own Dataset

```
[Your images folder: mydata/]
        │
        ▼
python dataset_tool.py
   --source=mydata/
   --dest=mydata.zip
   --transform=center-crop
   --width=256 --height=256
        │
        ▼
[mydata.zip]  ← training-ready archive
        │
        ▼
python train.py
   --outdir=training-runs
   --data=mydata.zip
   --gpus=1
   --cfg=auto        ← auto-selects best config
   --mirror=1        ← horizontal flip augmentation
        │
        ▼
[training-runs/00000-mydata-auto1/]
   ├── network-snapshot-000000.pkl  ← saved checkpoints
   ├── network-snapshot-001000.pkl
   ├── log.txt
   └── training_options.json
```

### 🌐 Pipeline F — Web UI

```
python server.py
        │
        ▼
Open browser: http://localhost:5000
        │
        ▼ (All pipelines A–D available via web interface)
[Generate] [Style Mix] [Discover] [Edit] [Interpolate] [Project]
        │
        ▼
Results displayed in-browser with live progress log
```

---

## 6. Module Reference

### `generate.py`

**Purpose:** Generate images from the pretrained network using integer seeds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--network` | str | required | Path or URL to `.pkl` model file |
| `--seeds` | range/list | — | Seeds to generate (e.g. `1,2,3` or `0-99`) |
| `--trunc` | float | 1.0 | Truncation ψ. Lower = more average/safe. Range: 0–1 |
| `--class` | int | None | Class label (only for conditional models like CIFAR-10) |
| `--noise-mode` | choice | `const` | `const` = deterministic, `random` = vary each run, `none` = no noise |
| `--projected-w` | str | None | Path to `.npz` projected W file (skips seed generation) |
| `--outdir` | str | required | Output directory |

**How it works internally:**
```python
z = np.random.RandomState(seed).randn(1, G.z_dim)   # ← seed → z vector
w = G.mapping(z_tensor, label)                        # ← z → w (style code)
img = G.synthesis(w, noise_mode=noise_mode)           # ← w → image
```

---

### `style_mixing.py`

**Purpose:** Create a grid of images where coarse styles (structure) come from row seeds and fine styles (color/texture) come from column seeds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--network` | str | required | Model path |
| `--rows` | list | required | Seeds for rows (coarse/structure donor) |
| `--cols` | list | required | Seeds for columns (fine/color donor) |
| `--styles` | range | `0-6` | Which synthesis layers to copy from cols |
| `--trunc` | float | 1.0 | Truncation psi |
| `--noise-mode` | choice | `const` | Noise mode |
| `--outdir` | str | required | Output directory |

**Style Layer Semantics:**

| Layers | Resolution | Controls |
|--------|-----------|----------|
| `0–3` | 4×4 – 32×32 | **Coarse**: Face shape, pose, hairstyle |
| `4–7` | 64×64 – 128×128 | **Medium**: Facial features, expression |
| `8–13` | 256×256 – 1024×1024 | **Fine**: Skin color, texture, fine details |

**How it works:**
```python
all_w = G.mapping(all_z, None)                # generate W for all seeds
w_dict = {seed: w for seed, w in ...}         # map seed → W vector

# For each (row, col) pair:
w = w_dict[row_seed].clone()
w[col_styles] = w_dict[col_seed][col_styles]  # swap selected layers
image = G.synthesis(w)                         # generate mixed image
```

---

### `explore_latent.py`

**Purpose:** Explore and manipulate the latent space using SeFa (Closed-Form Factorization).

Has **3 subcommands**:

#### `discover` — Find Editing Directions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--network` | required | Model path |
| `--seeds` | `100,200,300` | Seeds to demonstrate directions on |
| `--num-directions` | 10 | Number of top directions to find |
| `--strength` | 4.0 | Editing strength range (±strength) |
| `--trunc` | 0.7 | Truncation psi |
| `--steps` | 7 | Number of strength variations per direction |
| `--outdir` | required | Output directory |

**Outputs:** `discover_seed{N}.png` grid + `sefa_directions.npz`

#### `edit` — Apply a Direction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | required | Face to edit |
| `--direction` | 0 | Which discovered direction to apply (0 = most impactful) |
| `--strength` | `-5,5` | Strength range as `"min,max"` |
| `--steps` | 11 | Number of steps across range |
| `--layers` | `0-7` | W layers to modify |
| `--directions-file` | auto | Path to `sefa_directions.npz` |

**Layer Guide:**
- `0-3` → Coarse changes (pose, shape)
- `4-7` → Medium changes (features, expression)  
- `8-13` → Fine changes (color, texture)

#### `interpolate` — Morph Between Two Faces

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed-start` | required | Starting face seed |
| `--seed-end` | required | Ending face seed |
| `--frames` | 60 | Number of animation frames |
| `--mode` | `slerp` | `slerp` (smooth sphere interpolation) or `linear` |
| `--loop` | False | If set, morphs A→B→A (for looping GIFs) |
| `--trunc` | 0.7 | Truncation |

**SLERP (Spherical Linear Interpolation) vs Linear:**
- **SLERP**: Travels along a sphere in high-dim space → smoother, more natural transitions
- **Linear**: Direct straight-line interpolation → simpler, can look slightly unnatural at midpoints

---

### `projector.py`

**Purpose:** Find the W latent code that produces an image closest to your target photo.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--network` | required | Model path |
| `--target` | required | Your input image (any size/format) |
| `--num-steps` | 1000 | Optimization iterations (more = better quality) |
| `--seed` | 303 | Random seed for reproducible projection |
| `--save-video` | True | Save `.mp4` of optimization progress |
| `--outdir` | required | Output directory |

**Optimization Process:**
```
Initialize projected W ← mean of W space
For each step:
    Generate image from current W
    Compute LPIPS perceptual loss vs target (using VGG16)
    Compute noise regularization loss
    Adam optimizer step → update W
Save final W as projected_w.npz
```

**VGG16 is downloaded automatically** from NVIDIA CDN on first run.

---

### `dataset_tool.py`

**Purpose:** Convert your image collection into a training-ready format.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | required | Input: folder, zip, LMDB, CIFAR-10 `.tar.gz`, or MNIST `.gz` |
| `--dest` | required | Output: folder or `.zip` archive |
| `--max-images` | None | Limit number of images |
| `--transform` | None | `center-crop` or `center-crop-wide` |
| `--width` | None | Output width (pixels) |
| `--height` | None | Output height (pixels) |
| `--resize-filter` | `lanczos` | `lanczos` or `box` |

**Dataset Requirements:**
- Images must be **square** (same width and height)
- Dimensions must be a **power of 2** (64, 128, 256, 512, 1024)
- All images must have the **same resolution**

**Supported Input Formats:**
| Source | Auto-detected by |
|--------|-----------------|
| Image folder | Directory path (non-LMDB) |
| Zip archive | `.zip` extension |
| LSUN LMDB | Path ending in `_lmdb` |
| CIFAR-10 | File named `cifar-10-python.tar.gz` |
| MNIST | File named `train-images-idx3-ubyte.gz` |

---

### `train.py`

**Purpose:** Train a StyleGAN2-ADA model on your custom dataset.

**Key Parameters:**

| Category | Parameter | Description |
|----------|-----------|-------------|
| Required | `--data` | Path to training dataset (folder or zip) |
| Required | `--outdir` | Where to save training runs |
| GPUs | `--gpus` | Number of GPUs (must be power of 2) |
| Config | `--cfg` | `auto` (recommended), `stylegan2`, `paper256`, `paper512`, `paper1024`, `cifar` |
| Duration | `--kimg` | Total training kimgs (default: 25,000) |
| Augmentation | `--aug` | `ada` (default), `noaug`, `fixed` |
| Transfer | `--resume` | `ffhq256`, `ffhq512`, `ffhq1024`, `celebahq256`, or custom pkl |
| Snapshot | `--snap` | Save checkpoint every N ticks (default: 50) |
| Mirror | `--mirror` | Enable horizontal flip augmentation |

**Training Run Output:**
```
training-runs/
└── 00000-mydata-auto1/
    ├── network-snapshot-000000.pkl   ← initial checkpoint
    ├── network-snapshot-001000.pkl   ← @1000 kimgs
    ├── ...
    ├── log.txt                       ← full training log
    ├── training_options.json         ← exact config used
    └── stats.jsonl                   ← per-tick metrics
```

**Recommended Configs for Small Datasets:**

| Dataset Size | Suggested Config | Notes |
|-------------|-----------------|-------|
| < 1,000 images | `--cfg=auto --aug=ada` | ADA adapts automatically |
| 1,000–5,000 | `--cfg=auto --aug=ada --mirror=1` | Flip augmentation helps |
| Transfer learning | `--resume=ffhq256 --cfg=paper256` | Start from pretrained |

---

### `server.py`

**Purpose:** Flask-based HTTP API that runs all generation scripts in background threads and serves a Web UI.

**Start the server:**
```powershell
python server.py
# Open: http://localhost:5000
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the Web UI (`ui/index.html`) |
| `GET` | `/image/<path>` | Serve a generated image file |
| `GET` | `/api/job/<job_id>` | Poll job status + logs + result images |
| `POST` | `/api/generate` | Start random image generation |
| `POST` | `/api/stylemix` | Start style mixing |
| `POST` | `/api/discover` | Start latent direction discovery |
| `POST` | `/api/edit` | Start latent direction editing |
| `POST` | `/api/interpolate` | Start face morphing |
| `POST` | `/api/project` | Upload + project a custom image |
| `GET` | `/api/gallery/<folder>` | List images in `out/`, `style_mix/`, or `latent/` |

**Job System Architecture:**
```
POST /api/generate  →  new_job()  →  run_script() in background thread
                                           │
                                    subprocess.Popen(generate.py ...)
                                           │
                           Poll: GET /api/job/<id>  →  {status, log, images}
                                           │
                              status = "running" | "done" | "error"
```

---

## 7. Web UI Backend

The UI lives in `ui/`:

| File | Size | Purpose |
|------|------|---------|
| `index.html` | ~14.5 KB | Single-page app structure, tab navigation |
| `style.css` | ~11 KB | Dark theme, card layouts, responsive styles |
| `app.js` | ~16.5 KB | Tab switching, form submission, job polling, image gallery |

**Frontend Flow:**
```
User fills form → JS sends POST to API → Gets job_id
                     ↓
         Polls GET /api/job/<job_id> every 1.5s
                     ↓
         Shows real-time log output in UI
                     ↓
         On status="done": loads result images from /image/...
```

---

## 8. Output Folders Reference

| Folder | Created by | Contents |
|--------|-----------|----------|
| `out/` | `generate.py`, `projector.py` | `seed{N}.png`, `proj.png`, `proj.mp4`, `projected_w.npz`, `target.png` |
| `style_mix_results/` | `style_mixing.py` | `grid.png`, `{row}-{col}.png` individual mixes |
| `latent_explore/` | `explore_latent.py` | `discover_seed{N}.png`, `sefa_directions.npz`, `edit_strip_*.png`, `morph_{A}_to_{B}/` |
| `latent_explore/morph_{A}_to_{B}/` | `explore_latent.py interpolate` | `frame_{N:04d}.png`, `filmstrip_preview.png` |
| `training-runs/` | `train.py` | Checkpoint PKLs, logs, metrics |

---

## 9. Key Concepts Explained

### Truncation Trick (`--trunc`)

```
w_truncated = w_average + (w - w_average) × ψ

ψ = 1.0  → Full diversity, may look unusual
ψ = 0.7  → Balanced quality/diversity (recommended)
ψ = 0.5  → Very safe, very average-looking faces
ψ = 0.0  → Everyone looks the same (mean face)
```

### Noise Mode

| Mode | Behavior | Use Case |
|------|----------|----------|
| `const` | Same noise every run | Deterministic, reproducible |
| `random` | Different noise each time | Slight texture variation |
| `none` | No noise injected | Useful for debugging |

### Seeds

A **seed** is just an integer that initializes a random number generator. The **same seed always produces the same image** from the same model. Seeds allow you to:
- Reproduce results exactly
- Share specific interesting images by number
- Systematically explore the latent space

### FID Score (Fréchet Inception Distance)

Lower FID = better image quality. Measures how close the distribution of generated images is to real images.
- `< 10` — Very high quality
- `10–30` — Good quality
- `> 50` — Noticeable artifacts

---

## 10. Troubleshooting

### ❌ CUDA out of memory

```powershell
# Reduce batch size during training
python train.py --data=... --outdir=... --batch=8

# Or use fewer seeds when generating
python generate.py --seeds=1,2,3 --outdir=out  # instead of 1-100
```

### ❌ `No module named 'dnnlib'`

```powershell
# Make sure you're in the project directory
cd C:\Users\iamir\Downloads\gan-aca

# And virtual env is activated
.venv\Scripts\activate
```

### ❌ Custom CUDA ops fail to compile

```powershell
# Install Visual C++ Build Tools (Windows)
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or run with --fp32 to avoid CUDA kernel requirements
python train.py --data=... --outdir=... --fp32=True
```

### ❌ VGG16 download fails during projection

The projector auto-downloads VGG16 from NVIDIA's CDN. If it fails:
1. Check internet connectivity
2. Try a VPN if CDN is blocked in your region
3. Manually download and specify the path in `projector.py` line 60

### ❌ Server port already in use

```powershell
# Change port in server.py line 306:
app.run(host="0.0.0.0", port=5001, ...)  # change 5000 to any free port
```

### ❌ `projected_w.npz` shape mismatch

Generated `projected_w.npz` is model-specific. If you switch models, you need to re-project your image with the new model.

---

## 📎 Quick Reference Card

```powershell
# ── Generate ──────────────────────────────────────────────
python generate.py --network=network.pkl --seeds=1-10 --trunc=0.7 --outdir=out

# ── Style Mix ─────────────────────────────────────────────
python style_mixing.py --network=network.pkl --rows=85,100 --cols=55,821 --outdir=style_mix_results

# ── Discover Directions ───────────────────────────────────
python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=100,200

# ── Edit a Face ───────────────────────────────────────────
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=200 --direction=0

# ── Morph / Interpolate ───────────────────────────────────
python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=100 --seed-end=300 --frames=60

# ── Project Your Face ─────────────────────────────────────
python projector.py --network=network.pkl --target=my_face.png --outdir=out --num-steps=1000

# ── Web UI ────────────────────────────────────────────────
python server.py   # then open http://localhost:5000

# ── Convert Dataset ───────────────────────────────────────
python dataset_tool.py --source=myimages/ --dest=mydata.zip --transform=center-crop --width=256 --height=256

# ── Train ─────────────────────────────────────────────────
python train.py --outdir=training-runs --data=mydata.zip --gpus=1 --cfg=auto --mirror=1
```

---

*Documentation generated for StyleGAN2-ADA-PyTorch project at `C:\Users\iamir\Downloads\gan-aca`*  
*Generated: 2026-02-23*
