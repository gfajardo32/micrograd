"""Bake the trained weights into a single self-contained HTML page.

Reads model.json (written by train_digits.py) and samples.json, and writes
docs/index.html with everything inlined: no server, no network, no build
step. Open the file and it works.

Run:  python make_page.py
"""

import json
import os

MODEL = json.load(open("model.json"))
SAMPLES = json.load(open("samples.json"))

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draw a digit &mdash; micrograd</title>
<style>
  :root {
    color-scheme: light dark;
    --surface-0: #f4f4f2;
    --surface-1: #fcfcfb;
    --border:    #e3e3df;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #86857f;
    --series-1: #2a78d6;
    --series-1-wash: rgba(42,120,214,0.12);
    --ink: #0b0b0b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-0: #121211;
      --surface-1: #1a1a19;
      --border:    #302f2d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #8a8980;
      --series-1: #3987e5;
      --series-1-wash: rgba(57,135,229,0.16);
      --ink: #ffffff;
    }
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    padding: 32px 20px 56px;
    background: var(--surface-0);
    color: var(--text-primary);
    font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  .wrap { max-width: 940px; margin: 0 auto; }

  header { margin-bottom: 28px; }
  h1 { margin: 0 0 6px; font-size: 24px; font-weight: 620; letter-spacing: -0.01em; }
  .sub { margin: 0; color: var(--text-secondary); font-size: 14px; max-width: 62ch; }
  .sub code { font-size: 13px; background: var(--surface-1); padding: 1px 5px;
              border-radius: 4px; border: 1px solid var(--border); }

  .cols { display: grid; grid-template-columns: minmax(0,320px) minmax(0,1fr); gap: 24px; align-items: start; }
  @media (max-width: 760px) { .cols { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }

  .card h2 {
    margin: 0 0 14px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-muted);
  }

  #pad {
    width: 100%; aspect-ratio: 1; display: block;
    background: var(--surface-0);
    border: 1px solid var(--border);
    border-radius: 8px;
    touch-action: none; cursor: crosshair;
  }

  .row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }

  button {
    font: inherit; font-size: 13px; padding: 7px 13px;
    background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 7px; cursor: pointer;
  }
  button:hover { background: var(--surface-0); }
  button:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }

  .seen { display: flex; align-items: center; gap: 12px; margin-top: 16px;
          padding-top: 14px; border-top: 1px solid var(--border); }
  #grid { image-rendering: pixelated; width: 72px; height: 72px;
          border: 1px solid var(--border); border-radius: 5px; background: var(--surface-0); }
  .seen p { margin: 0; font-size: 12.5px; color: var(--text-secondary); }

  .hero { display: flex; align-items: baseline; gap: 14px; margin-bottom: 4px; }
  .hero .big { font-size: 76px; line-height: 0.9; font-weight: 600;
               font-variant-numeric: tabular-nums; letter-spacing: -0.03em; }
  .hero .meta { font-size: 13px; color: var(--text-secondary); }
  .hero .dim { color: var(--text-muted); }

  table.chart { width: 100%; border-collapse: collapse; margin-top: 18px; }
  table.chart th { text-align: left; font-size: 11px; font-weight: 600;
                   letter-spacing: 0.05em; text-transform: uppercase;
                   color: var(--text-muted); padding-bottom: 8px; font-variant-numeric: tabular-nums; }
  table.chart th.num, table.chart td.num { text-align: right; }
  table.chart td { padding: 3px 0; vertical-align: middle; }
  td.lab { width: 26px; font-variant-numeric: tabular-nums; color: var(--text-secondary);
           font-size: 14px; padding-right: 10px; }
  tr.top td.lab { color: var(--text-primary); font-weight: 640; }
  td.bar { width: 100%; }
  .track { height: 14px; background: var(--series-1-wash); border-radius: 4px; position: relative; }
  .fill { height: 14px; background: var(--series-1);
          border-radius: 0 4px 4px 0; width: 0; transition: width 90ms linear; }
  td.val { width: 56px; padding-left: 10px; font-size: 13px;
           font-variant-numeric: tabular-nums; color: var(--text-secondary); }
  tr.top td.val { color: var(--text-primary); font-weight: 600; }

  .note { margin: 22px 0 0; font-size: 12.5px; color: var(--text-muted); max-width: 74ch; }
  .note strong { color: var(--text-secondary); font-weight: 600; }

  .samples { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .samples button { padding: 6px 10px; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Draw a digit</h1>
    <p class="sub">
      Every number below is computed by a 1,210-parameter network trained with a
      scalar autograd engine written from scratch &mdash; <code>engine.py</code>,
      about 120 lines. No PyTorch, no NumPy, no maths library. The weights are
      baked into this file and the forward pass runs in your browser.
    </p>
  </header>

  <div class="cols">

    <div class="card">
      <h2>Input</h2>
      <canvas id="pad" width="336" height="336"></canvas>
      <div class="row">
        <button id="clear">Clear</button>
        <button id="undo">Undo stroke</button>
      </div>

      <div class="seen">
        <canvas id="grid" width="8" height="8"></canvas>
        <p>What the network sees: 8&times;8, 64 pixels, the same shape as its
           training data.</p>
      </div>

      <div class="row" style="margin-top:16px;display:block">
        <h2 style="margin-bottom:8px">Or try a real held-out image</h2>
        <div class="samples" id="samples"></div>
      </div>
    </div>

    <div class="card">
      <h2>Prediction</h2>
      <div class="hero">
        <span class="big" id="guess">&mdash;</span>
        <span class="meta" id="caption">draw something on the left</span>
      </div>

      <table class="chart">
        <caption class="sr-only" style="position:absolute;left:-9999px">
          Output strength per digit
        </caption>
        <thead>
          <tr>
            <th colspan="2">Output strength by digit</th>
            <th class="num">score</th>
          </tr>
        </thead>
        <tbody id="bars"></tbody>
      </table>

      <p class="note">
        <strong>On the numbers.</strong> The ten outputs are <code>tanh</code>
        values in (&minus;1, 1), rescaled here to 0&ndash;100. They are not
        probabilities and do not sum to 100 &mdash; each output was trained
        independently to say &ldquo;yes this digit&rdquo; or &ldquo;no it
        isn&rsquo;t&rdquo;. The prediction is simply whichever output is largest.
      </p>
      <p class="note">
        <strong>If it misreads your drawing</strong>, that is honest. It learned
        from 1,437 scanned digits written by adults in the 1990s on a
        pressure-sensitive tablet, at a resolution of 64 pixels. Draw thick,
        centred and boxy and it does much better. Real accuracy on images it
        never saw during training is printed at the bottom.
      </p>
    </div>
  </div>

  <p class="note" id="scorecard"></p>
</div>

<script>
const MODEL   = __MODEL__;
const SAMPLES = __SAMPLES__;
const TEST_ACC = __TEST_ACC__;

/* ---- the network's forward pass, same maths as engine.py ---------------- */
function forward(x) {
  let a = x;
  for (const layer of MODEL.layers) {
    const out = [];
    for (let n = 0; n < layer.b.length; n++) {
      let act = layer.b[n];
      const w = layer.w[n];
      for (let i = 0; i < w.length; i++) act += w[i] * a[i];
      out.push(Math.tanh(act));
    }
    a = out;
  }
  return a;
}

/* ---- drawing ------------------------------------------------------------ */
const pad = document.getElementById('pad');
const ctx = pad.getContext('2d', { willReadFrequently: true });
const SIZE = pad.width;
let strokes = [], current = null, drawing = false;

function repaint() {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.lineWidth = 26;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim();
  for (const s of strokes) {
    if (s.length < 2) {
      if (s.length === 1) {
        ctx.beginPath(); ctx.arc(s[0][0], s[0][1], 13, 0, 7);
        ctx.fillStyle = ctx.strokeStyle; ctx.fill();
      }
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(s[0][0], s[0][1]);
    for (let i = 1; i < s.length; i++) ctx.lineTo(s[i][0], s[i][1]);
    ctx.stroke();
  }
}

function pos(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * SIZE / r.width, (e.clientY - r.top) * SIZE / r.height];
}

pad.addEventListener('pointerdown', e => {
  pad.setPointerCapture(e.pointerId);
  drawing = true; current = [pos(e)]; strokes.push(current); repaint(); run();
});
pad.addEventListener('pointermove', e => {
  if (!drawing) return;
  current.push(pos(e)); repaint(); run();
});
addEventListener('pointerup', () => { drawing = false; });

document.getElementById('clear').onclick = () => { strokes = []; repaint(); run(); };
document.getElementById('undo').onclick  = () => { strokes.pop(); repaint(); run(); };

/* ---- turn the drawing into 64 numbers ----------------------------------- */
/* The training images were built by centring a digit in a 32x32 bitmap and
   counting the on-pixels in each 4x4 block, giving 64 values from 0 to 16.
   We reproduce that exactly, so the network sees the format it learned on. */
function toFeatures() {
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const on = (x, y) => px[(y * SIZE + x) * 4 + 3] > 40;

  // bounding box of the ink
  let x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++)
      if (on(x, y)) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  if (x1 < 0) return null;   // nothing drawn

  // scale the box to fit a 28x28 area, preserving aspect ratio, centred in 32x32
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const scale = 28 / Math.max(bw, bh);
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = 32;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  const dw = bw * scale, dh = bh * scale;
  tctx.drawImage(pad, x0, y0, bw, bh, (32 - dw) / 2, (32 - dh) / 2, dw, dh);

  // count on-pixels in each 4x4 block -> 0..16, then rescale to 0..1
  const t = tctx.getImageData(0, 0, 32, 32).data;
  const feats = [];
  for (let by = 0; by < 8; by++)
    for (let bx = 0; bx < 8; bx++) {
      let count = 0;
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++)
          if (t[(((by * 4 + y) * 32) + bx * 4 + x) * 4 + 3] > 40) count++;
      feats.push(count / 16);
    }
  return feats;
}

/* ---- output ------------------------------------------------------------- */
const bars = document.getElementById('bars');
const rows = [];
for (let d = 0; d < 10; d++) {
  const tr = document.createElement('tr');
  tr.innerHTML = '<td class="lab">' + d + '</td>' +
                 '<td class="bar"><div class="track"><div class="fill"></div></div></td>' +
                 '<td class="val num">&mdash;</td>';
  bars.appendChild(tr);
  rows.push({ tr, fill: tr.querySelector('.fill'), val: tr.querySelector('.val') });
}

const gridCtx = document.getElementById('grid').getContext('2d');

function drawGrid(feats) {
  const img = gridCtx.createImageData(8, 8);
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  for (let i = 0; i < 64; i++) {
    const v = feats ? feats[i] : 0;
    const shade = dark ? Math.round(255 * v) : Math.round(255 * (1 - v));
    img.data[i*4] = img.data[i*4+1] = img.data[i*4+2] = shade;
    img.data[i*4+3] = 255;
  }
  gridCtx.putImageData(img, 0, 0);
}

function show(feats) {
  drawGrid(feats);
  if (!feats) {
    document.getElementById('guess').textContent = '\\u2014';
    document.getElementById('caption').innerHTML = '<span class="dim">draw something on the left</span>';
    rows.forEach(r => { r.fill.style.width = '0%'; r.val.innerHTML = '&mdash;'; r.tr.classList.remove('top'); });
    return;
  }
  const outs = forward(feats);
  const scores = outs.map(o => (o + 1) / 2 * 100);
  let best = 0;
  for (let i = 1; i < 10; i++) if (scores[i] > scores[best]) best = i;

  const sorted = [...scores].sort((a, b) => b - a);
  const margin = sorted[0] - sorted[1];

  document.getElementById('guess').textContent = best;
  document.getElementById('caption').textContent =
    margin > 25 ? 'confident' : margin > 10 ? 'fairly sure' : 'not sure at all';

  rows.forEach((r, d) => {
    r.fill.style.width = scores[d].toFixed(1) + '%';
    r.val.textContent = scores[d].toFixed(1);
    r.tr.classList.toggle('top', d === best);
  });
}

function run() { show(toFeatures()); }

/* ---- sample buttons ----------------------------------------------------- */
const holder = document.getElementById('samples');
Object.keys(SAMPLES).sort().forEach(d => {
  const b = document.createElement('button');
  b.textContent = d;
  b.title = 'a real ' + d + ' the network never trained on';
  b.onclick = () => { strokes = []; repaint(); show(SAMPLES[d]); };
  holder.appendChild(b);
});

document.getElementById('scorecard').innerHTML =
  '<strong>Scorecard.</strong> ' + TEST_ACC.toFixed(2) + '% correct on ' +
  '360 held-out images the network never saw during training. ' +
  'The ten buttons above are drawn from that same held-out set.';

repaint();
show(null);
</script>
</body>
</html>
"""


def build(test_acc, out="docs/index.html"):
    # docs/ is what GitHub Pages serves, so the demo gets a live URL for free
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    html = (HTML
            .replace("__MODEL__", json.dumps(MODEL))
            .replace("__SAMPLES__", json.dumps(SAMPLES))
            .replace("__TEST_ACC__", repr(float(test_acc))))
    with open(out, "w") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    import sys
    acc = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print("wrote", build(acc))
