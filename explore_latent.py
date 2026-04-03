"""
Latent Space Exploration for StyleGAN2-ADA-PyTorch
===================================================
Uses SeFa (Closed-Form Factorization) to discover meaningful editing
directions in the generator's latent space.

Modes:
  1. discover  — Find and visualize top editing directions
  2. edit      — Apply a specific direction to a face
  3. interpolate — Smooth morph between two seeds (video frames)

Examples:
    # Discover top 10 editing directions
    python explore_latent.py discover --network=network.pkl --outdir=latent_explore --seeds=100,200,300

    # Edit a face along direction 0 (often age/pose)
    python explore_latent.py edit --network=network.pkl --outdir=latent_explore --seed=200 --direction=0 --strength=-5,5

    # Morph between two faces (generates frames for video)
    python explore_latent.py interpolate --network=network.pkl --outdir=latent_explore --seed-start=100 --seed-end=300 --frames=60
"""

import os
import re
import sys
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# ============================================================================
# SeFa: Closed-Form Factorization
# ============================================================================

def sefa_directions(G, num_directions=10, layer_index='all'):
    """
    Discover meaningful latent directions using SeFa.

    SeFa analyzes the weight matrix of the first layer of the synthesis network.
    The eigenvectors of (A^T * A) correspond to the most impactful directions
    in the latent space — these often align with semantic attributes like
    age, pose, smile, lighting, etc.

    Args:
        G: Generator network
        num_directions: Number of top directions to extract
        layer_index: Which synthesis layer(s) to analyze ('all' or int)

    Returns:
        eigvecs: Eigenvectors (directions) of shape [z_dim, num_directions]
        eigvals: Eigenvalues (importance) of shape [num_directions]
    """
    # Collect weight matrices from the synthesis network's modulation layers
    modulate_weights = []

    for name, param in G.named_parameters():
        # Look for the affine transformation weights in synthesis layers
        # These are the style modulation weights that control generation
        if 'affine' in name and 'weight' in name and param.ndim == 2:
            modulate_weights.append(param.detach().cpu().numpy())

    if not modulate_weights:
        # Fallback: try mapping network weights
        for name, param in G.named_parameters():
            if 'mapping' in name and 'weight' in name and param.ndim == 2:
                modulate_weights.append(param.detach().cpu().numpy())

    if not modulate_weights:
        raise RuntimeError("Could not find suitable weight matrices for SeFa analysis")

    if layer_index == 'all':
        # Stack all layer weights
        W = np.concatenate(modulate_weights, axis=0)
    else:
        W = modulate_weights[min(layer_index, len(modulate_weights) - 1)]

    print(f"  Analyzing weight matrix of shape {W.shape} from {len(modulate_weights)} layers...")

    # Compute eigenvectors of W^T @ W
    # These eigenvectors represent the most significant directions
    WtW = W.T @ W
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)

    # Sort by eigenvalue magnitude (largest = most impactful)
    sort_idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx[:num_directions]]
    eigenvectors = eigenvectors[:, sort_idx[:num_directions]]

    return eigenvectors, eigenvalues


def generate_image(G, w, noise_mode='const'):
    """Generate image from W latent vector."""
    img = G.synthesis(w, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img[0].cpu().numpy()


def seed_to_w(G, seed, device, truncation_psi=0.7):
    """Convert a seed to a W latent vector."""
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    w = G.mapping(z, None)
    w_avg = G.mapping.w_avg
    w = w_avg + (w - w_avg) * truncation_psi
    return w


# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
def cli():
    """Latent Space Exploration for StyleGAN2-ADA"""
    pass


# --------------------------------------------------------------------------
# DISCOVER: Find and visualize top editing directions
# --------------------------------------------------------------------------
@cli.command()
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--seeds', type=str, default='100,200,300', help='Seeds to demonstrate directions on')
@click.option('--num-directions', type=int, default=10, help='Number of directions to discover')
@click.option('--strength', type=float, default=4.0, help='Editing strength')
@click.option('--trunc', type=float, default=0.7, help='Truncation psi')
@click.option('--steps', type=int, default=7, help='Number of edit steps per direction')
def discover(network_pkl, outdir, seeds, num_directions, strength, trunc, steps):
    """Discover and visualize the top latent editing directions."""

    device = torch.device('cuda')
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Parse seeds
    seed_list = [int(s.strip()) for s in seeds.split(',')]

    # Discover directions using SeFa
    print('\n🔍 Discovering latent directions with SeFa...')
    eigvecs, eigvals = sefa_directions(G, num_directions=num_directions)

    print(f'\n📊 Top {num_directions} direction eigenvalues (importance):')
    for i, val in enumerate(eigvals):
        bar = '█' * int(abs(val) / abs(eigvals[0]) * 30)
        print(f'  Direction {i:2d}: {val:12.2f}  {bar}')

    # Generate grid for each seed
    for seed in seed_list:
        print(f'\n🎨 Generating exploration grid for seed {seed}...')
        w_base = seed_to_w(G, seed, device, trunc)

        # Create grid: rows = directions, cols = strength variations
        strengths = np.linspace(-strength, strength, steps)
        W = G.img_resolution
        H = G.img_resolution

        # Grid: (num_directions rows) x (steps cols + 1 label column)
        canvas = PIL.Image.new('RGB', (W * (steps + 1), H * num_directions), 'black')

        for dir_idx in range(num_directions):
            direction = torch.from_numpy(eigvecs[:, dir_idx]).float().to(device)

            for step_idx, s in enumerate(strengths):
                # Apply direction in W space
                w_edit = w_base.clone()
                w_edit[:, :8, :] += s * direction  # Apply to early/mid layers

                img = generate_image(G, w_edit)
                pil_img = PIL.Image.fromarray(img, 'RGB')

                # Add label for center (original) image
                x_pos = W * (step_idx + 1)
                y_pos = H * dir_idx
                canvas.paste(pil_img, (x_pos, y_pos))

            # Create label image for first column
            label_img = PIL.Image.new('RGB', (W, H), (30, 30, 30))
            # Draw text manually with simple approach
            from PIL import ImageDraw
            draw = ImageDraw.Draw(label_img)
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = ImageFont.load_default()

            text = f"Dir {dir_idx}"
            importance = abs(eigvals[dir_idx]) / abs(eigvals[0]) * 100
            draw.text((10, H//2 - 40), text, fill='white', font=font)
            draw.text((10, H//2), f"{importance:.0f}%", fill=(100, 200, 255), font=font)
            draw.text((10, H//2 + 40), f"←{-strength:.0f}  →+{strength:.0f}", fill=(150, 150, 150), font=font)
            canvas.paste(label_img, (0, H * dir_idx))

        grid_path = f'{outdir}/discover_seed{seed}.png'
        canvas.save(grid_path)
        print(f'  ✅ Saved: {grid_path}')

    # Save direction data for later use
    np.savez(f'{outdir}/sefa_directions.npz', eigvecs=eigvecs, eigvals=eigvals)
    print(f'\n💾 Saved directions to {outdir}/sefa_directions.npz')
    print(f'\n🎯 Next: Edit a specific direction with:')
    print(f'   python explore_latent.py edit --network={network_pkl} --outdir={outdir} --seed=200 --direction=0')


# --------------------------------------------------------------------------
# EDIT: Apply a specific direction to a face
# --------------------------------------------------------------------------
@cli.command()
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--seed', type=int, required=True, help='Seed for the base face')
@click.option('--direction', type=int, default=0, help='Direction index to apply')
@click.option('--strength', type=str, default='-5,5', help='Strength range (min,max)')
@click.option('--steps', type=int, default=11, help='Number of steps')
@click.option('--trunc', type=float, default=0.7, help='Truncation psi')
@click.option('--layers', type=str, default='0-7', help='Which W layers to edit (e.g., 0-3 for coarse, 4-7 for mid, 8-13 for fine)')
@click.option('--directions-file', type=str, default=None, help='Path to sefa_directions.npz (auto-detected)')
def edit(network_pkl, outdir, seed, direction, strength, steps, trunc, layers, directions_file):
    """Edit a face along a specific latent direction."""

    device = torch.device('cuda')
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Parse layer range
    layer_match = re.match(r'^(\d+)-(\d+)$', layers)
    if layer_match:
        layer_start, layer_end = int(layer_match.group(1)), int(layer_match.group(2))
    else:
        layer_start = layer_end = int(layers)

    # Parse strength
    s_min, s_max = [float(x) for x in strength.split(',')]

    # Load or compute directions
    if directions_file and os.path.exists(directions_file):
        data = np.load(directions_file)
        eigvecs = data['eigvecs']
        print(f'  Loaded directions from {directions_file}')
    elif os.path.exists(f'{outdir}/sefa_directions.npz'):
        data = np.load(f'{outdir}/sefa_directions.npz')
        eigvecs = data['eigvecs']
        print(f'  Loaded directions from {outdir}/sefa_directions.npz')
    else:
        print('  Computing SeFa directions...')
        eigvecs, _ = sefa_directions(G)

    dir_vec = torch.from_numpy(eigvecs[:, direction]).float().to(device)

    print(f'\n🎭 Editing seed {seed} along direction {direction}')
    print(f'   Strength: {s_min} → {s_max} ({steps} steps)')
    print(f'   Layers: {layer_start}-{layer_end}')

    w_base = seed_to_w(G, seed, device, trunc)
    strengths = np.linspace(s_min, s_max, steps)

    W = G.img_resolution
    H = G.img_resolution
    images = []

    for i, s in enumerate(strengths):
        w_edit = w_base.clone()
        w_edit[:, layer_start:layer_end+1, :] += s * dir_vec

        img = generate_image(G, w_edit)
        images.append(img)

        # Save individual image
        pil_img = PIL.Image.fromarray(img, 'RGB')
        pil_img.save(f'{outdir}/edit_seed{seed}_dir{direction}_s{s:+.1f}.png')

    # Create horizontal strip
    strip = PIL.Image.new('RGB', (W * steps, H + 60), (20, 20, 20))
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(strip)

    for i, (img, s) in enumerate(zip(images, strengths)):
        strip.paste(PIL.Image.fromarray(img, 'RGB'), (W * i, 0))
        label = f'{s:+.1f}'
        color = (100, 255, 100) if abs(s) < 0.1 else (200, 200, 200)
        draw.text((W * i + W//2 - 20, H + 15), label, fill=color, font=font)

    strip_path = f'{outdir}/edit_strip_seed{seed}_dir{direction}.png'
    strip.save(strip_path)
    print(f'\n  ✅ Saved strip: {strip_path}')
    print(f'  ✅ Saved {steps} individual images')

    # Tip
    print(f'\n💡 Tips:')
    print(f'   --layers 0-3   → Coarse edits (pose, shape, structure)')
    print(f'   --layers 4-7   → Medium edits (features, expression)')
    print(f'   --layers 8-13  → Fine edits (color, texture, details)')


# --------------------------------------------------------------------------
# INTERPOLATE: Smooth morph between two seeds
# --------------------------------------------------------------------------
@cli.command()
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--seed-start', type=int, required=True, help='Starting seed')
@click.option('--seed-end', type=int, required=True, help='Ending seed')
@click.option('--frames', type=int, default=60, help='Number of interpolation frames')
@click.option('--trunc', type=float, default=0.7, help='Truncation psi')
@click.option('--mode', type=click.Choice(['linear', 'slerp']), default='slerp', help='Interpolation mode')
@click.option('--loop', is_flag=True, help='Loop back to start')
def interpolate(network_pkl, outdir, seed_start, seed_end, frames, trunc, mode, loop):
    """Generate smooth face morphing frames between two seeds."""

    device = torch.device('cuda')
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    morph_dir = f'{outdir}/morph_{seed_start}_to_{seed_end}'
    os.makedirs(morph_dir, exist_ok=True)

    w_start = seed_to_w(G, seed_start, device, trunc)
    w_end = seed_to_w(G, seed_end, device, trunc)

    def slerp(w1, w2, t):
        """Spherical linear interpolation — smoother than linear for high-dim vectors."""
        w1_flat = w1.reshape(-1)
        w2_flat = w2.reshape(-1)

        # Normalize
        w1_norm = w1_flat / w1_flat.norm()
        w2_norm = w2_flat / w2_flat.norm()

        # Angle between vectors
        omega = torch.acos(torch.clamp(torch.dot(w1_norm, w2_norm), -1, 1))

        if omega.abs() < 1e-6:
            return w1 * (1 - t) + w2 * t

        sin_omega = torch.sin(omega)
        result = (torch.sin((1 - t) * omega) / sin_omega) * w1_flat + \
                 (torch.sin(t * omega) / sin_omega) * w2_flat
        return result.reshape(w1.shape)

    if loop:
        # Generate round-trip: start → end → start
        ts = np.concatenate([
            np.linspace(0, 1, frames // 2, endpoint=False),
            np.linspace(1, 0, frames // 2, endpoint=True)
        ])
    else:
        ts = np.linspace(0, 1, frames)

    print(f'\n🎬 Morphing seed {seed_start} → seed {seed_end} ({len(ts)} frames)')
    print(f'   Mode: {mode} | Loop: {loop}')

    for i, t in enumerate(ts):
        if mode == 'slerp':
            w = slerp(w_start, w_end, t)
        else:
            w = w_start * (1 - t) + w_end * t

        img = generate_image(G, w)
        PIL.Image.fromarray(img, 'RGB').save(f'{morph_dir}/frame_{i:04d}.png')

        if (i + 1) % 10 == 0 or i == 0:
            print(f'   Frame {i+1}/{len(ts)} (t={t:.3f})')

    print(f'\n  ✅ Saved {len(ts)} frames to {morph_dir}/')

    # Create filmstrip preview
    preview_count = min(10, len(ts))
    preview_indices = np.linspace(0, len(ts) - 1, preview_count, dtype=int)
    W = G.img_resolution
    H = G.img_resolution
    filmstrip = PIL.Image.new('RGB', (W * preview_count, H), 'black')
    for j, idx in enumerate(preview_indices):
        frame = PIL.Image.open(f'{morph_dir}/frame_{idx:04d}.png')
        filmstrip.paste(frame, (W * j, 0))
    filmstrip_path = f'{morph_dir}/filmstrip_preview.png'
    filmstrip.save(filmstrip_path)
    print(f'  ✅ Saved filmstrip preview: {filmstrip_path}')

    print(f'\n💡 To make a video (requires ffmpeg):')
    print(f'   ffmpeg -framerate 30 -i {morph_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {outdir}/morph.mp4')


# ============================================================================

if __name__ == "__main__":
    cli()
