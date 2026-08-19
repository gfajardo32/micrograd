// Check that the JavaScript forward pass baked into the HTML page reproduces
// the Python engine's predictions exactly. Run: node verify_page.js
const fs = require('fs');

const MODEL = JSON.parse(fs.readFileSync('model.json', 'utf8'));
const CASES = JSON.parse(fs.readFileSync('parity_cases.json', 'utf8'));

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

let worst = 0, mismatches = 0;
for (const c of CASES) {
  const js = forward(c.x);
  for (let i = 0; i < 10; i++) worst = Math.max(worst, Math.abs(js[i] - c.outs[i]));
  const argmax = js.indexOf(Math.max(...js));
  if (argmax !== c.pred) mismatches++;
}
console.log(`cases:            ${CASES.length}`);
console.log(`argmax mismatches: ${mismatches}`);
console.log(`worst output diff: ${worst.toExponential(3)}`);
process.exit(mismatches === 0 && worst < 1e-12 ? 0 : 1);
