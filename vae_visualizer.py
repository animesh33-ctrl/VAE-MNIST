"""
VAE Interactive Visualizer
Run: python vae_visualizer.py
Open: http://localhost:5000
"""

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template_string, request, jsonify
import base64
import io
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 20
CKPT_PATH  = "vae_outputs/vae_best.pth"

# ─── MODEL (same arch as training) ────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, 1, 28, 28)).shape[1]
        self.fc_mu      = nn.Linear(flat_size, latent_dim)
        self.fc_log_var = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), 128, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, lv = self.encoder(x)
        std = torch.exp(0.5 * lv)
        z   = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, lv


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded | Device: {DEVICE}")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def tensor_to_b64(img_tensor):
    """Convert (1,1,28,28) tensor → base64 PNG string for browser."""
    arr = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L').resize((280, 280), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def grid_to_b64(tensors, nrow=8):
    """Multiple tensors → grid PNG → base64."""
    n   = len(tensors)
    cols = nrow
    rows = (n + cols - 1) // cols
    grid = Image.new('L', (cols * 28, rows * 28), 0)
    for i, t in enumerate(tensors):
        arr = (t.squeeze().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
        grid.paste(img, ((i % cols) * 28, (i // cols) * 28))
    grid = grid.resize((cols * 56, rows * 56), Image.NEAREST)
    buf  = io.BytesIO()
    grid.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VAE Visualizer</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --panel: #0f0f1a;
    --border: #1e1e3a;
    --accent: #00ff88;
    --accent2: #ff006e;
    --text: #c8c8e0;
    --dim: #555570;
    --font-mono: 'Share Tech Mono', monospace;
    --font-ui: 'Rajdhani', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-ui);
    min-height: 100vh;
    padding: 24px;
  }
  h1 {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    color: var(--accent);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .subtitle {
    font-size: 0.85rem;
    color: var(--dim);
    font-family: var(--font-mono);
    margin-bottom: 28px;
  }
  .layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 20px;
    max-width: 1100px;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px;
  }
  .panel-title {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
  }
  .tab {
    flex: 1;
    padding: 8px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--dim);
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.15s;
    text-transform: uppercase;
  }
  .tab.active {
    background: var(--accent);
    color: #000;
    border-color: var(--accent);
  }
  .tab:hover:not(.active) { border-color: var(--accent); color: var(--accent); }

  /* sliders */
  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .dim-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
    width: 36px;
    flex-shrink: 0;
  }
  input[type=range] {
    flex: 1;
    -webkit-appearance: none;
    height: 2px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }
  .val-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--accent);
    width: 36px;
    text-align: right;
    flex-shrink: 0;
  }
  .btn {
    width: 100%;
    padding: 10px;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 3px;
    cursor: pointer;
    text-transform: uppercase;
    transition: all 0.15s;
    margin-top: 12px;
  }
  .btn:hover { background: var(--accent); color: #000; }
  .btn.danger { border-color: var(--accent2); color: var(--accent2); }
  .btn.danger:hover { background: var(--accent2); color: #fff; }

  /* output */
  .output-panel { display: flex; flex-direction: column; gap: 20px; }
  .img-box {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 180px;
    position: relative;
    overflow: hidden;
  }
  .img-box img { image-rendering: pixelated; max-width: 100%; }
  .img-label {
    position: absolute;
    top: 8px; left: 10px;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--dim);
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .placeholder {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--dim);
    letter-spacing: 2px;
  }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .info-bar {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--dim);
    padding: 8px 0;
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }
  .info-bar span { color: var(--accent); }
  select {
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 6px 8px;
    width: 100%;
    margin-bottom: 10px;
    outline: none;
    cursor: pointer;
  }
  .section { margin-bottom: 16px; }
  .section-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  #loading {
    display: none;
    position: fixed;
    top: 12px; right: 12px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 2px;
    animation: blink 0.6s infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
</style>
</head>
<body>
<div id="loading">GENERATING...</div>
<h1>VAE // LATENT EXPLORER</h1>
<p class="subtitle">conv-vae · mnist · latent_dim=20 · trained from scratch</p>

<div class="layout">
  <!-- LEFT PANEL -->
  <div class="panel">
    <div class="panel-title">Control</div>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('random')">Random</button>
      <button class="tab" onclick="switchTab('sliders')">Sliders</button>
      <button class="tab" onclick="switchTab('interp')">Morph</button>
      <button class="tab" onclick="switchTab('grid')">Grid</button>
    </div>

    <!-- RANDOM TAB -->
    <div id="tab-random">
      <div class="section">
        <div class="section-title">Count</div>
        <input type="range" min="1" max="64" value="64" id="rand-count"
               oninput="document.getElementById('rand-count-val').textContent=this.value">
        <div class="slider-row">
          <span class="dim-label">n =</span>
          <span id="rand-count-val" class="val-label" style="width:auto">64</span>
        </div>
      </div>
      <button class="btn" onclick="generateRandom()">[ Generate ]</button>
      <button class="btn danger" onclick="randomizeAndGenerate()">[ Randomize + Go ]</button>
    </div>

    <!-- SLIDERS TAB -->
    <div id="tab-sliders" style="display:none">
      <div id="sliders-container"></div>
      <button class="btn" onclick="generateFromSliders()">[ Decode ]</button>
      <button class="btn danger" onclick="resetSliders()">[ Reset to Zero ]</button>
    </div>

    <!-- INTERP TAB -->
    <div id="tab-interp" style="display:none">
      <div class="section">
        <div class="section-title">Steps</div>
        <div class="slider-row">
          <span class="dim-label">n</span>
          <input type="range" min="4" max="20" value="10" id="interp-steps"
                 oninput="document.getElementById('interp-steps-val').textContent=this.value">
          <span class="val-label" id="interp-steps-val">10</span>
        </div>
      </div>
      <button class="btn" onclick="sampleZ1()">[ Sample Z1 ]</button>
      <button class="btn" onclick="sampleZ2()">[ Sample Z2 ]</button>
      <button class="btn" onclick="interpolate()">[ Interpolate ]</button>
      <div class="info-bar" id="interp-info">z1: <span>—</span> | z2: <span>—</span></div>
    </div>

    <!-- GRID TAB -->
    <div id="tab-grid" style="display:none">
      <div class="section">
        <div class="section-title">Latent Dimension to sweep</div>
        <select id="grid-dim">
          {% for i in range(20) %}
          <option value="{{i}}">dim_{{i}}</option>
          {% endfor %}
        </select>
        <div class="section-title">Range</div>
        <div class="slider-row">
          <span class="dim-label">-val</span>
          <input type="range" min="1" max="5" value="3" step="0.5" id="grid-range"
                 oninput="document.getElementById('grid-range-val').textContent=this.value">
          <span class="val-label" id="grid-range-val">3</span>
        </div>
      </div>
      <button class="btn" onclick="sweepDim()">[ Sweep Dimension ]</button>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="output-panel">
    <div class="panel">
      <div class="panel-title">Output</div>
      <div class="img-box" id="main-output" style="min-height:320px">
        <span class="img-label">decoded output</span>
        <span class="placeholder">[ press generate ]</span>
      </div>
      <div class="info-bar" id="status-bar">
        status: <span>idle</span> | device: <span>{{device}}</span> | latent_dim: <span>20</span>
      </div>
    </div>
  </div>
</div>

<script>
let currentTab = 'random';
let z1_vec = null, z2_vec = null;

// ── TAB SWITCH ────────────────────────────────────────────────────────────────
function switchTab(tab) {
  ['random','sliders','interp','grid'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===tab ? 'block' : 'none';
  });
  document.querySelectorAll('.tab').forEach((b,i) => {
    b.classList.toggle('active', ['random','sliders','interp','grid'][i]===tab);
  });
  currentTab = tab;
  if (tab === 'sliders' && document.getElementById('sliders-container').children.length === 0) {
    buildSliders();
  }
}

// ── SLIDERS ───────────────────────────────────────────────────────────────────
function buildSliders() {
  const c = document.getElementById('sliders-container');
  for (let i = 0; i < 20; i++) {
    c.innerHTML += `
      <div class="slider-row">
        <span class="dim-label">z${String(i).padStart(2,'0')}</span>
        <input type="range" min="-3" max="3" step="0.05" value="0"
               id="z${i}" oninput="document.getElementById('v${i}').textContent=parseFloat(this.value).toFixed(2)">
        <span class="val-label" id="v${i}">0.00</span>
      </div>`;
  }
}

function getSliderValues() {
  return Array.from({length:20}, (_,i) => parseFloat(document.getElementById('z'+i).value));
}

function resetSliders() {
  for (let i=0;i<20;i++) {
    document.getElementById('z'+i).value = 0;
    document.getElementById('v'+i).textContent = '0.00';
  }
}

// ── API CALLS ─────────────────────────────────────────────────────────────────
function showLoading(on) {
  document.getElementById('loading').style.display = on ? 'block' : 'none';
}

function setStatus(msg) {
  document.getElementById('status-bar').innerHTML =
    `status: <span>${msg}</span> | device: <span>{{device}}</span> | latent_dim: <span>20</span>`;
}

function showImage(b64, label='decoded output') {
  const box = document.getElementById('main-output');
  box.innerHTML = `<span class="img-label">${label}</span><img src="data:image/png;base64,${b64}">`;
}

async function post(url, data) {
  showLoading(true);
  const r = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(data)
  });
  const j = await r.json();
  showLoading(false);
  return j;
}

async function generateRandom() {
  const n = parseInt(document.getElementById('rand-count').value);
  setStatus('generating random z...');
  const j = await post('/generate_random', {n});
  showImage(j.img, `random generation · n=${n}`);
  setStatus(`done · ${n} samples`);
}

async function randomizeAndGenerate() {
  const n = parseInt(document.getElementById('rand-count').value);
  setStatus('randomizing...');
  const j = await post('/generate_random', {n});
  showImage(j.img, `random generation · n=${n}`);
  setStatus(`done · ${n} samples`);
}

async function generateFromSliders() {
  const z = getSliderValues();
  setStatus('decoding z...');
  const j = await post('/decode_z', {z});
  showImage(j.img, 'slider decode · z→x');
  setStatus('decoded');
}

async function sampleZ1() {
  const j = await post('/sample_z', {});
  z1_vec = j.z;
  document.getElementById('interp-info').innerHTML =
    `z1: <span>sampled ✓</span> | z2: <span>${z2_vec?'sampled ✓':'—'}</span>`;
  setStatus('z1 sampled');
}

async function sampleZ2() {
  const j = await post('/sample_z', {});
  z2_vec = j.z;
  document.getElementById('interp-info').innerHTML =
    `z1: <span>${z1_vec?'sampled ✓':'—'}</span> | z2: <span>sampled ✓</span>`;
  setStatus('z2 sampled');
}

async function interpolate() {
  if (!z1_vec || !z2_vec) { alert('Sample Z1 and Z2 first'); return; }
  const steps = parseInt(document.getElementById('interp-steps').value);
  setStatus('interpolating...');
  const j = await post('/interpolate', {z1: z1_vec, z2: z2_vec, steps});
  showImage(j.img, `interpolation · ${steps} steps`);
  setStatus(`interpolation done · ${steps} frames`);
}

async function sweepDim() {
  const dim   = parseInt(document.getElementById('grid-dim').value);
  const range = parseFloat(document.getElementById('grid-range').value);
  setStatus(`sweeping dim_${dim}...`);
  const j = await post('/sweep_dim', {dim, range});
  showImage(j.img, `dim_${dim} sweep · -${range} → +${range}`);
  setStatus(`sweep done · dim_${dim}`);
}
</script>
</body>
</html>
"""

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, device=str(DEVICE))

@app.route("/generate_random", methods=["POST"])
def generate_random():
    n = request.json.get("n", 64)
    with torch.no_grad():
        z   = torch.randn(n, LATENT_DIM).to(DEVICE)
        imgs = model.decode(z)
    b64 = grid_to_b64([imgs[i] for i in range(n)], nrow=8)
    return jsonify({"img": b64})

@app.route("/decode_z", methods=["POST"])
def decode_z():
    z_vals = request.json["z"]
    z = torch.tensor([z_vals], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        img = model.decode(z)
    b64 = tensor_to_b64(img)
    return jsonify({"img": b64})

@app.route("/sample_z", methods=["POST"])
def sample_z():
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    return jsonify({"z": z.squeeze().cpu().tolist()})

@app.route("/interpolate", methods=["POST"])
def interpolate():
    data  = request.json
    z1    = torch.tensor([data["z1"]], dtype=torch.float32).to(DEVICE)
    z2    = torch.tensor([data["z2"]], dtype=torch.float32).to(DEVICE)
    steps = data.get("steps", 10)
    with torch.no_grad():
        imgs = [model.decode((1-a)*z1 + a*z2)
                for a in torch.linspace(0, 1, steps).to(DEVICE)]
    b64 = grid_to_b64(imgs, nrow=steps)
    return jsonify({"img": b64})

@app.route("/sweep_dim", methods=["POST"])
def sweep_dim():
    data  = request.json
    dim   = data["dim"]
    rng   = data.get("range", 3.0)
    steps = 10
    with torch.no_grad():
        imgs = []
        for val in torch.linspace(-rng, rng, steps):
            z = torch.zeros(1, LATENT_DIM).to(DEVICE)
            z[0, dim] = val
            imgs.append(model.decode(z))
    b64 = grid_to_b64(imgs, nrow=steps)
    return jsonify({"img": b64})

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("VAE Visualizer running!")
    print("Open browser → http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
