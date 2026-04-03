# gan-aca

`gan-aca` is a StyleGAN2-ADA based project with added local experimentation files for generation, latent exploration, and UI/server work.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Use

Generate images:

```bash
python generate.py --outdir=out --seeds=1-8 --network=network.pkl
```

Run the local server:

```bash
python server.py
```

## How It Works

- core GAN training and inference come from the StyleGAN2-ADA codebase
- `training/`, `metrics/`, `torch_utils/`, and `dnnlib/` support training and evaluation
- local folders like `ui/` and related scripts support project-specific experimentation
