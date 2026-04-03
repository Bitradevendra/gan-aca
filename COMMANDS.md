# 🎨 StyleGAN2-ADA-PyTorch — Complete Command Reference

> **Working Directory:** `C:\Users\iamir\Downloads\stylegan2-ada-pytorch`
> **Local Model:** `network.pkl` ✅ (already downloaded)

---

## 📁 Setup — Navigate to Project Folder

```powershell
cd C:\Users\iamir\Downloads\stylegan2-ada-pytorch
```

---

## 1️⃣ Generate Random Images (`generate.py`)

```powershell
# Generate 4 specific seeds
python generate.py --network=network.pkl --seeds=1,2,3,4 --outdir=out

# Generate a range of seeds (0 to 9 = 10 images)
python generate.py --network=network.pkl --seeds=0-9 --outdir=out

# With truncation (0.5–0.8 = safe/realistic, 1.0 = full diversity)
python generate.py --network=network.pkl --seeds=100-110 --trunc=0.7 --outdir=out

# With random noise mode (slightly different textures each run)
python generate.py --network=network.pkl --seeds=42,69,420 --trunc=0.8 --noise-mode=random --outdir=out

# With constant noise (deterministic — same result every run)
python generate.py --network=network.pkl --seeds=500-510 --trunc=1.0 --noise-mode=const --outdir=out
```

---

## 2️⃣ Style Mixing (`style_mixing.py`)

> Combines **coarse features** (structure/pose) from row seeds with **fine features** (color/texture) from column seeds.
> Output: Individual images + a `grid.png` showing the full mixing matrix.

```powershell
# Basic style mix — default style layers (0-6)
python style_mixing.py --network=network.pkl --rows=85,100,75 --cols=55,821,293 --outdir=style_mix_results

# Mix only COARSE styles (pose, shape) — layers 0-3
python style_mixing.py --network=network.pkl --rows=85,100,75 --cols=55,821,293 --styles=0-3 --outdir=style_mix_results

# Mix only FINE styles (color, texture) — layers 8-13
python style_mixing.py --network=network.pkl --rows=85,100,75 --cols=55,821,293 --styles=8-13 --outdir=style_mix_results

# With truncation for cleaner faces
python style_mixing.py --network=network.pkl --rows=1,2,3,4,5 --cols=10,20,30,40 --trunc=0.7 --outdir=style_mix_results
```

### 🗂️ Style Layer Guide

| Layers  | Controls                        |
|---------|---------------------------------|
| `0-3`   | Coarse — pose, face shape, hair |
| `4-7`   | Medium — facial features, expression |
| `8-13`  | Fine — color, texture, skin details |

---

## 3️⃣ Latent Space Exploration (`explore_latent.py`)

> This script has **3 subcommands**: `discover`, `edit`, `interpolate`

---

### 🔍 3a. DISCOVER — Find Meaningful Edit Directions

> Runs SeFa (Closed-Form Factorization) to find the most impactful directions in latent space.
> Output: `discover_seed{N}.png` grids + `sefa_directions.npz` (reusable in `edit`).

```powershell
# Discover top 10 directions, visualized on 3 seeds
python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=100,200,300

# More directions, higher strength, custom seeds
python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=42,99,512 --num-directions=15 --strength=6.0 --steps=9

# Faster preview (fewer directions, fewer steps)
python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=1,2 --num-directions=5 --steps=5
```

---

### 🎭 3b. EDIT — Apply a Direction to a Face

> Apply a discovered direction to change specific attributes of a face.
> Output: Individual edit images + `edit_strip_seed{N}_dir{D}.png` horizontal strip.

```powershell
# Edit seed 200 along direction 0 (run discover first!)
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=200 --direction=0

# Custom strength range (-8 to +8) with more steps
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=200 --direction=0 --strength="-8,8" --steps=13

# Edit COARSE layers only (pose/structure changes)
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=100 --direction=1 --layers=0-3

# Edit FINE layers only (color/texture changes)
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=100 --direction=2 --layers=8-13

# Edit MEDIUM layers (expression/features)
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=100 --direction=3 --layers=4-7
```

---

### 🎬 3c. INTERPOLATE — Morph Between Two Faces

> Generates smooth animation frames morphing between two seeds.
> Output: Frame PNGs + `filmstrip_preview.png`.

```powershell
# Smooth morph from seed 100 to seed 300 (60 frames, slerp)
python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=100 --seed-end=300 --frames=60

# Linear interpolation (simpler, slightly faster)
python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=42 --seed-end=999 --frames=30 --mode=linear

# Loop animation (A → B → A, great for GIFs)
python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=100 --seed-end=500 --frames=60 --loop

# Convert frames to video using FFmpeg
ffmpeg -framerate 30 -i latent_explore/morph_100_to_300/frame_%04d.png -c:v libx264 -pix_fmt yuv420p latent_explore/morph.mp4
```

---

## 4️⃣ Project Your Custom Image into Latent Space (`projector.py`)

> Encodes **your own photo** into the GAN's W latent space so you can then edit it.

```powershell
# Step 1 — Project your image (replace my_face.png with your image path)
python projector.py --network=network.pkl --target=my_face.png --outdir=out --num-steps=1000

# Faster projection (less accurate, 500 steps)
python projector.py --network=network.pkl --target=my_face.png --outdir=out --num-steps=500

# Step 2 — Generate image FROM the projected W vector
python generate.py --network=network.pkl --projected-w=out/projected_w.npz --outdir=out
```

> **Output Files:**
> - `projected_w.npz` — the latent code for your image
> - `proj.png` — reconstructed image
> - `proj.mp4` — optimization progress video

---

## 🔁 Full Recommended Workflow (Custom Image → Edit → Animate)

```powershell
# 1. Project your custom image into latent space
python projector.py --network=network.pkl --target=my_face.png --outdir=out --num-steps=1000

# 2. Discover meaningful edit directions
python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=100,200,300

# 3. Edit along interesting directions
python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=200 --direction=0 --strength="-6,6" --layers=0-7

# 4. Create a morph animation between two seeds
python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=100 --seed-end=200 --frames=60 --loop

# 5. Render animation to video
ffmpeg -framerate 30 -i latent_explore/morph_100_to_200/frame_%04d.png -c:v libx264 -pix_fmt yuv420p latent_explore/morph.mp4
```

---

## 💡 Pro Tips

| Tip | Details |
|-----|---------|
| `--trunc=0.7` | Cleaner, more photorealistic faces (recommended) |
| `--trunc=1.0` | Maximum diversity (can look unusual/weird) |
| Reuse directions | The `sefa_directions.npz` saved by `discover` avoids recomputing each time |
| Direction priority | **Direction 0** is always the most impactful, direction 1 next, etc. |
| Coarse vs Fine | Use layers `0-3` for big structural changes, `8-13` for subtle color/texture edits |
| Custom image size | Projector auto-resizes your image — no need to pre-crop it manually |

---

## 📂 Output Folder Reference

| Folder | Contents |
|--------|----------|
| `out/` | Random generated images, projected images |
| `style_mix_results/` | Style mixing grids and individual mixed images |
| `latent_explore/` | Direction discovery grids, edit strips, morph frames |
| `latent_explore/morph_X_to_Y/` | Interpolation frames + filmstrip preview |
