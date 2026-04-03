/* =============================================
   StyleGAN2 Studio — Frontend Logic
   ============================================= */

const API = "http://localhost:5000";
let pollTimers = {};

// ─── Tab Switching ─────────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
        if (btn.dataset.tab === "gallery") loadGallery("out", document.querySelector(".gtab.active"));
    });
});

// ─── Slider sync ──────────────────────────────────────────────────────────
function syncVal(sliderId, valId) {
    document.getElementById(valId).textContent = document.getElementById(sliderId).value;
}

// ─── Job Polling ───────────────────────────────────────────────────────────
function pollJob(jobId, statusEl, logEl, resultsEl, onDone) {
    if (pollTimers[jobId]) clearInterval(pollTimers[jobId]);
    pollTimers[jobId] = setInterval(async () => {
        try {
            const res = await fetch(`${API}/api/job/${jobId}`);
            const job = await res.json();

            // Update log
            if (job.log && job.log.length) {
                logEl.classList.remove("hidden");
                logEl.textContent = job.log.slice(-30).join("\n");
                logEl.scrollTop = logEl.scrollHeight;
            }

            if (job.status === "running") {
                statusEl.className = "status-box running";
                statusEl.innerHTML = `<div class="spinner"></div> Running… please wait`;
            } else if (job.status === "done") {
                clearInterval(pollTimers[jobId]);
                statusEl.className = "status-box done";
                statusEl.innerHTML = `✅ Done! ${job.images.length} image(s) generated.`;
                if (job.images && job.images.length) {
                    renderImages(job.images, resultsEl);
                    if (onDone) onDone(job);
                } else {
                    statusEl.innerHTML += ` <small>(Check log for details)</small>`;
                }
            } else if (job.status === "error") {
                clearInterval(pollTimers[jobId]);
                statusEl.className = "status-box error";
                statusEl.innerHTML = `❌ Error: ${job.error || "Unknown error. Check log."}`;
            }
        } catch (e) {
            clearInterval(pollTimers[jobId]);
            statusEl.className = "status-box error";
            statusEl.innerHTML = `❌ Cannot reach server. Is server.py running?`;
        }
    }, 1500);
}

function startJob(statusEl, logEl, resultsEl) {
    statusEl.classList.remove("hidden");
    statusEl.className = "status-box running";
    statusEl.innerHTML = `<div class="spinner"></div> Starting job…`;
    logEl.classList.remove("hidden");
    logEl.textContent = "";
    resultsEl.innerHTML = "";
}

// ─── Render Images ─────────────────────────────────────────────────────────
function renderImages(urls, container) {
    container.innerHTML = "";
    urls.forEach(url => {
        const card = document.createElement("div");
        card.className = "img-card";
        const name = url.split("/").pop();
        card.innerHTML = `
      <img src="${API}${url}?t=${Date.now()}" alt="${name}" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22><rect fill=%22%23181c27%22 width=%22200%22 height=%22200%22/><text x=%2250%%22 y=%2250%%22 fill=%22%234a5280%22 text-anchor=%22middle%22 dy=%22.3em%22>No preview</text></svg>'" />
      <div class="img-label">${name}</div>
    `;
        card.onclick = () => openLightbox(`${API}${url}?t=${Date.now()}`, name);
        container.appendChild(card);
    });
}

// ─── Lightbox ──────────────────────────────────────────────────────────────
function openLightbox(src, name) {
    document.getElementById("lb-img").src = src;
    document.getElementById("lb-name").textContent = name;
    document.getElementById("lightbox").classList.remove("hidden");
}
function closeLightbox() {
    document.getElementById("lightbox").classList.add("hidden");
}
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

// ─── 1. GENERATE ───────────────────────────────────────────────────────────
async function runGenerate() {
    const statusEl = document.getElementById("gen-status");
    const logEl = document.getElementById("gen-log");
    const resEl = document.getElementById("gen-results");
    startJob(statusEl, logEl, resEl);

    const body = {
        seeds: document.getElementById("gen-seeds").value.trim(),
        trunc: parseFloat(document.getElementById("gen-trunc").value),
        noise_mode: document.getElementById("gen-noise").value
    };

    try {
        const res = await fetch(`${API}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error";
        statusEl.innerHTML = "❌ Cannot reach server. Make sure <code>python server.py</code> is running.";
    }
}

// ─── 2. STYLE MIX ──────────────────────────────────────────────────────────
async function runStyleMix() {
    const statusEl = document.getElementById("mix-status");
    const logEl = document.getElementById("mix-log");
    const resEl = document.getElementById("mix-results");
    startJob(statusEl, logEl, resEl);

    const body = {
        rows: document.getElementById("mix-rows").value.trim(),
        cols: document.getElementById("mix-cols").value.trim(),
        styles: document.getElementById("mix-styles").value,
        trunc: parseFloat(document.getElementById("mix-trunc").value)
    };

    try {
        const res = await fetch(`${API}/api/stylemix`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error"; statusEl.innerHTML = "❌ Cannot reach server.";
    }
}

// ─── 3a. DISCOVER ──────────────────────────────────────────────────────────
async function runDiscover() {
    const statusEl = document.getElementById("disc-status");
    const logEl = document.getElementById("disc-log");
    const resEl = document.getElementById("disc-results");
    startJob(statusEl, logEl, resEl);

    const body = {
        seeds: document.getElementById("disc-seeds").value.trim(),
        num_directions: parseInt(document.getElementById("disc-dirs").value),
        strength: parseFloat(document.getElementById("disc-strength").value),
        steps: parseInt(document.getElementById("disc-steps").value),
        trunc: parseFloat(document.getElementById("disc-trunc").value)
    };

    try {
        const res = await fetch(`${API}/api/discover`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error"; statusEl.innerHTML = "❌ Cannot reach server.";
    }
}

// ─── 3b. EDIT ─────────────────────────────────────────────────────────────
async function runEdit() {
    const statusEl = document.getElementById("edit-status");
    const logEl = document.getElementById("edit-log");
    const resEl = document.getElementById("edit-results");
    startJob(statusEl, logEl, resEl);

    const body = {
        seed: parseInt(document.getElementById("edit-seed").value),
        direction: parseInt(document.getElementById("edit-dir").value),
        strength: document.getElementById("edit-strength").value.trim(),
        steps: parseInt(document.getElementById("edit-steps").value),
        layers: document.getElementById("edit-layers").value,
        trunc: parseFloat(document.getElementById("edit-trunc").value)
    };

    try {
        const res = await fetch(`${API}/api/edit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error"; statusEl.innerHTML = "❌ Cannot reach server.";
    }
}

// ─── 3c. INTERPOLATE ───────────────────────────────────────────────────────
async function runInterpolate() {
    const statusEl = document.getElementById("interp-status");
    const logEl = document.getElementById("interp-log");
    const resEl = document.getElementById("interp-results");
    startJob(statusEl, logEl, resEl);

    const body = {
        seed_start: parseInt(document.getElementById("interp-start").value),
        seed_end: parseInt(document.getElementById("interp-end").value),
        frames: parseInt(document.getElementById("interp-frames").value),
        mode: document.getElementById("interp-mode").value,
        loop: document.getElementById("interp-loop").checked,
        trunc: parseFloat(document.getElementById("interp-trunc").value)
    };

    try {
        const res = await fetch(`${API}/api/interpolate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error"; statusEl.innerHTML = "❌ Cannot reach server.";
    }
}

// ─── 4. PROJECT ────────────────────────────────────────────────────────────
function previewUpload(input) {
    const preview = document.getElementById("upload-preview");
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = e => {
            preview.innerHTML = `<img src="${e.target.result}" style="max-height:150px;border-radius:8px;" />`;
        };
        reader.readAsDataURL(input.files[0]);
        // Update zone to show filename
        const zone = document.getElementById("upload-zone");
        zone.querySelector("img") || (zone.innerHTML += `<p style="color:#8892b0;margin-top:8px;font-size:12px">${input.files[0].name}</p>`);
    }
}

async function runProject() {
    const statusEl = document.getElementById("proj-status");
    const logEl = document.getElementById("proj-log");
    const resEl = document.getElementById("proj-results");
    const fileInput = document.getElementById("proj-file");

    if (!fileInput.files || !fileInput.files[0]) {
        statusEl.classList.remove("hidden");
        statusEl.className = "status-box error";
        statusEl.innerHTML = "❌ Please select an image file first.";
        return;
    }

    startJob(statusEl, logEl, resEl);

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    formData.append("num_steps", document.getElementById("proj-steps").value);

    try {
        const res = await fetch(`${API}/api/project`, { method: "POST", body: formData });
        const { job_id, error } = await res.json();
        if (error) { statusEl.className = "status-box error"; statusEl.innerHTML = `❌ ${error}`; return; }
        pollJob(job_id, statusEl, logEl, resEl);
    } catch (e) {
        statusEl.className = "status-box error"; statusEl.innerHTML = "❌ Cannot reach server.";
    }
}

// ─── GALLERY ───────────────────────────────────────────────────────────────
async function loadGallery(folder, btn) {
    document.querySelectorAll(".gtab").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");

    const grid = document.getElementById("gallery-grid");
    grid.innerHTML = `<div style="color:var(--text2);grid-column:1/-1;padding:20px 0;">Loading…</div>`;

    try {
        const res = await fetch(`${API}/api/gallery/${folder}`);
        const images = await res.json();
        if (!images.length) {
            grid.innerHTML = `<div style="color:var(--text3);grid-column:1/-1;padding:20px 0;">No images found in this folder yet. Generate some first!</div>`;
            return;
        }
        renderImages(images.map(i => i.url.replace(API, "")), grid);
    } catch (e) {
        grid.innerHTML = `<div style="color:var(--red);grid-column:1/-1;">❌ Cannot reach server.</div>`;
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
    // Init all slider displays
    ["gen-trunc", "mix-trunc", "disc-strength", "disc-trunc", "edit-trunc", "interp-trunc"].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            const valId = id + "-val";
            if (document.getElementById(valId)) {
                document.getElementById(valId).textContent = el.value;
            }
        }
    });

    // Drag and drop on upload zone
    const zone = document.getElementById("upload-zone");
    if (zone) {
        zone.addEventListener("dragover", e => { e.preventDefault(); zone.style.borderColor = "var(--accent)"; });
        zone.addEventListener("dragleave", () => zone.style.borderColor = "");
        zone.addEventListener("drop", e => {
            e.preventDefault();
            zone.style.borderColor = "";
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) {
                const dt = new DataTransfer();
                dt.items.add(file);
                document.getElementById("proj-file").files = dt.files;
                previewUpload(document.getElementById("proj-file"));
            }
        });
    }
});
