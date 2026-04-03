"""
StyleGAN2-ADA PyTorch — Web UI Backend
Run with: python server.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import subprocess
import threading
import glob
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask import Response
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_PKL = os.path.join(BASE_DIR, "network.pkl")
PYTHON = sys.executable

app = Flask(__name__, static_folder="ui", static_url_path="")
if CORS_AVAILABLE:
    from flask_cors import CORS
    CORS(app)

# --------------------------------------------------------------------------
# Track running jobs
# --------------------------------------------------------------------------
jobs = {}
job_lock = threading.Lock()

def new_job(name):
    job_id = f"{name}_{datetime.now().strftime('%H%M%S%f')[:12]}"
    with job_lock:
        jobs[job_id] = {"status": "running", "log": [], "error": None, "images": []}
    return job_id

def run_script(job_id, cmd, outdir, image_glob="*.png"):
    """Run a script in a thread, capture output, collect result images."""
    def _run():
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in proc.stdout:
                line = line.rstrip()
                with job_lock:
                    jobs[job_id]["log"].append(line)
            proc.wait()
            # Collect generated images
            images = sorted(glob.glob(os.path.join(outdir, image_glob)))
            web_paths = ["/image/" + os.path.relpath(p, BASE_DIR).replace("\\", "/") for p in images]
            with job_lock:
                if proc.returncode == 0:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["images"] = web_paths
                else:
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = "Script exited with non-zero code"
        except Exception as e:
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# --------------------------------------------------------------------------
# Serve UI
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("ui", "index.html")

@app.route("/image/<path:filepath>")
def serve_image(filepath):
    full = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(full):
        return jsonify({"error": "Image not found"}), 404
    return send_file(full, mimetype="image/png")

# --------------------------------------------------------------------------
# Job Status
# --------------------------------------------------------------------------
@app.route("/api/job/<job_id>")
def job_status(job_id):
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

# --------------------------------------------------------------------------
# 1. Generate Random Images
# --------------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    seeds    = data.get("seeds", "1,2,3,4")
    trunc    = float(data.get("trunc", 0.7))
    noise    = data.get("noise_mode", "const")
    outdir   = os.path.join(BASE_DIR, "out")
    os.makedirs(outdir, exist_ok=True)

    # Validate seeds
    seeds = re.sub(r"[^0-9,\-]", "", str(seeds)).strip(",")
    if not seeds:
        return jsonify({"error": "Invalid seeds"}), 400

    job_id = new_job("generate")
    cmd = [
        PYTHON, "generate.py",
        f"--network={NETWORK_PKL}",
        f"--seeds={seeds}",
        f"--trunc={trunc}",
        f"--noise-mode={noise}",
        f"--outdir={outdir}"
    ]
    run_script(job_id, cmd, outdir, "seed*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# 2. Style Mixing
# --------------------------------------------------------------------------
@app.route("/api/stylemix", methods=["POST"])
def api_stylemix():
    data = request.json or {}
    rows   = re.sub(r"[^0-9,\-]", "", str(data.get("rows", "85,100,75"))).strip(",")
    cols   = re.sub(r"[^0-9,\-]", "", str(data.get("cols", "55,821,293"))).strip(",")
    styles = re.sub(r"[^0-9,\-]", "", str(data.get("styles", "0-6"))).strip(",")
    trunc  = float(data.get("trunc", 0.7))
    outdir = os.path.join(BASE_DIR, "style_mix_results")
    os.makedirs(outdir, exist_ok=True)

    job_id = new_job("stylemix")
    cmd = [
        PYTHON, "style_mixing.py",
        f"--network={NETWORK_PKL}",
        f"--rows={rows}",
        f"--cols={cols}",
        f"--styles={styles}",
        f"--trunc={trunc}",
        f"--outdir={outdir}"
    ]
    run_script(job_id, cmd, outdir, "*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# 3a. Latent Discover
# --------------------------------------------------------------------------
@app.route("/api/discover", methods=["POST"])
def api_discover():
    data    = request.json or {}
    seeds   = re.sub(r"[^0-9,]", "", str(data.get("seeds", "100,200,300")))
    n_dirs  = int(data.get("num_directions", 10))
    strength= float(data.get("strength", 4.0))
    steps   = int(data.get("steps", 7))
    trunc   = float(data.get("trunc", 0.7))
    outdir  = os.path.join(BASE_DIR, "latent_explore")
    os.makedirs(outdir, exist_ok=True)

    job_id = new_job("discover")
    cmd = [
        PYTHON, "explore_latent.py", "discover",
        f"--network={NETWORK_PKL}",
        f"--outdir={outdir}",
        f"--seeds={seeds}",
        f"--num-directions={n_dirs}",
        f"--strength={strength}",
        f"--steps={steps}",
        f"--trunc={trunc}"
    ]
    run_script(job_id, cmd, outdir, "discover_*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# 3b. Latent Edit
# --------------------------------------------------------------------------
@app.route("/api/edit", methods=["POST"])
def api_edit():
    data      = request.json or {}
    seed      = int(data.get("seed", 200))
    direction = int(data.get("direction", 0))
    strength  = str(data.get("strength", "-5,5"))
    steps     = int(data.get("steps", 11))
    layers    = str(data.get("layers", "0-7"))
    trunc     = float(data.get("trunc", 0.7))
    outdir    = os.path.join(BASE_DIR, "latent_explore")
    os.makedirs(outdir, exist_ok=True)

    job_id = new_job("edit")
    cmd = [
        PYTHON, "explore_latent.py", "edit",
        f"--network={NETWORK_PKL}",
        f"--outdir={outdir}",
        f"--seed={seed}",
        f"--direction={direction}",
        f"--strength={strength}",
        f"--steps={steps}",
        f"--layers={layers}",
        f"--trunc={trunc}"
    ]
    run_script(job_id, cmd, outdir, f"edit_seed{seed}_dir{direction}_*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# 3c. Interpolate
# --------------------------------------------------------------------------
@app.route("/api/interpolate", methods=["POST"])
def api_interpolate():
    data       = request.json or {}
    seed_start = int(data.get("seed_start", 100))
    seed_end   = int(data.get("seed_end", 300))
    frames     = int(data.get("frames", 30))
    mode       = data.get("mode", "slerp")
    loop       = bool(data.get("loop", False))
    trunc      = float(data.get("trunc", 0.7))
    outdir     = os.path.join(BASE_DIR, "latent_explore")
    os.makedirs(outdir, exist_ok=True)

    job_id = new_job("interp")
    cmd = [
        PYTHON, "explore_latent.py", "interpolate",
        f"--network={NETWORK_PKL}",
        f"--outdir={outdir}",
        f"--seed-start={seed_start}",
        f"--seed-end={seed_end}",
        f"--frames={frames}",
        f"--mode={mode}",
        f"--trunc={trunc}"
    ]
    if loop:
        cmd.append("--loop")
    morph_dir = os.path.join(outdir, f"morph_{seed_start}_to_{seed_end}")
    run_script(job_id, cmd, morph_dir, "frame_*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# 4. Project Custom Image
# --------------------------------------------------------------------------
@app.route("/api/project", methods=["POST"])
def api_project():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    target_path = os.path.join(BASE_DIR, "out", "target_input.png")
    os.makedirs(os.path.join(BASE_DIR, "out"), exist_ok=True)
    f.save(target_path)

    steps  = int(request.form.get("num_steps", 500))
    outdir = os.path.join(BASE_DIR, "out")

    job_id = new_job("project")
    cmd = [
        PYTHON, "projector.py",
        f"--network={NETWORK_PKL}",
        f"--target={target_path}",
        f"--num-steps={steps}",
        "--save-video=False",
        f"--outdir={outdir}"
    ]
    run_script(job_id, cmd, outdir, "proj*.png")
    return jsonify({"job_id": job_id})

# --------------------------------------------------------------------------
# Gallery — list all images in a folder
# --------------------------------------------------------------------------
@app.route("/api/gallery/<folder>")
def gallery(folder):
    folder_map = {
        "out": os.path.join(BASE_DIR, "out"),
        "style_mix": os.path.join(BASE_DIR, "style_mix_results"),
        "latent": os.path.join(BASE_DIR, "latent_explore"),
    }
    folder_path = folder_map.get(folder)
    if not folder_path or not os.path.exists(folder_path):
        return jsonify([])

    images = []
    for ext in ("*.png", "*.jpg"):
        for p in sorted(glob.glob(os.path.join(folder_path, ext)), key=os.path.getmtime, reverse=True):
            rel = os.path.relpath(p, BASE_DIR).replace("\\", "/")
            images.append({
                "url": f"/image/{rel}",
                "name": os.path.basename(p),
                "size": os.path.getsize(p)
            })
    return jsonify(images[:60])  # max 60

# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  🎨 StyleGAN2 Web UI")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
