# gan-aca

`gan-aca` is a StyleGAN2-ADA based project with additional local tooling for generation, latent exploration, and UI experimentation.

## Overview

The repository combines the StyleGAN2-ADA training stack with project-specific scripts, generated outputs, and a lightweight server/UI layer for experimentation.

## Project Structure

```text
gan-aca/
|-- training/
|-- metrics/
|-- torch_utils/
|-- dnnlib/
|-- ui/
|-- server.py
|-- generate.py
|-- train.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.7+
- NVIDIA GPU recommended
- CUDA toolkit compatible with the installed PyTorch build

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running The Project

Generate images:

```bash
python generate.py --outdir=out --seeds=1-8 --network=network.pkl
```

Run the local server:

```bash
python server.py
```

Train with your dataset:

```bash
python train.py --outdir=training-runs --data=path\to\dataset.zip --gpus=1
```

## How It Works

- the StyleGAN2-ADA core handles training and inference
- `training/`, `metrics/`, `torch_utils/`, and `dnnlib/` support the GAN pipeline
- `generate.py`, `projector.py`, and `style_mixing.py` provide generation and exploration tools
- `ui/` and `server.py` support local experimentation workflows
