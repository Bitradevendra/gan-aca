# GAN-ACA

A style-driven generative image lab built on StyleGAN2-ADA, extended with local tooling for experimentation, exploration, and visual iteration.

## Why This Repo Is Interesting

`gan-aca` is not just a training dump. It is a working image-generation sandbox: training scripts, latent exploration, generation utilities, a server, and a UI layer all orbit the same GAN core.

## What It Does

- trains and fine-tunes StyleGAN2-ADA models
- generates images from saved network checkpoints
- supports latent-space experimentation and style mixing
- exposes a lightweight backend for web-based generation workflows

## Project Structure

```text
gan-aca/
|-- training/
|-- metrics/
|-- torch_utils/
|-- dnnlib/
|-- ui/
|-- server.py
|-- train.py
|-- generate.py
|-- projector.py
|-- style_mixing.py
`-- README.md
```

## Requirements

- Python 3.7+
- NVIDIA GPU recommended
- CUDA-compatible PyTorch environment

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

Generate sample outputs:

```bash
python generate.py --outdir=out --seeds=1-8 --network=network.pkl
```

Start the local UI backend:

```bash
python server.py
```

Train a model:

```bash
python train.py --outdir=training-runs --data=path\to\dataset.zip --gpus=1
```

## How It Works

- StyleGAN2-ADA provides the generation and training backbone.
- `training/`, `metrics/`, `torch_utils/`, and `dnnlib/` carry the heavy lifting.
- `generate.py`, `projector.py`, and `style_mixing.py` handle output, projection, and experimentation.
- `server.py` and `ui/` provide a more approachable interface for interacting with the project.

## Why Someone Would Care

If you are exploring visual identity, synthetic datasets, generative art, or model-guided concept iteration, this repo has more hands-on range than a bare training implementation.
