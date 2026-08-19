# micrograd

A scalar autograd engine and a small neural net library, built from scratch.

**[Try it live](https://gfajardo32.github.io/micrograd/)** — draw a digit with your
mouse and watch a 1,210-parameter network read it. 95.83% on held-out images.

## Credit

This is not original work. It is a learning exercise, typed out and worked
through while following Andrej Karpathy's lecture
[**The spelled-out intro to neural networks and backpropagation: building
micrograd**](https://www.youtube.com/watch?v=VMj-3S1tku0), the first video in
his [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
series.

The design, the API and most of the implementation are his. The original project
lives at [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
and is MIT licensed. Anyone wanting to *use* a scalar autograd engine should go
there rather than here.

### Who wrote what

The engine — `engine.py` and `nn.py` — I typed out and debugged myself while
following the lecture. The bugs I hit on the way and how I found them are
written up under **The training loop** below; that debugging is where the
learning actually happened.

The digit classifier, the browser demo, and the initialisation fix
(`train_digits.py`, `make_page.py`, `docs/index.html`, `w_scale='auto'`) were
built with AI assistance. I've documented what each part does and why, and can
explain any of it, but I did not write it unaided and it would be dishonest to
imply otherwise.

## Layout

| File | Contains | Depends on |
| --- | --- | --- |
| `engine.py` | `Value` — the autograd engine | nothing |
| `nn.py` | `Neuron`, `Layer`, `MLP` | `engine` |
| `train.py` | the 4-example toy problem from the video | `nn` |
| `draw.py` | `draw_dot` graph rendering (optional) | `graphviz` |
| `train_digits.py` | handwritten digit classifier | `nn`, `digits.csv` |
| `make_page.py` | bakes the trained weights into the demo page | `model.json` |
| `docs/index.html` | draw a digit, watch it predict | nothing |
| `verify_page.js` | checks the JS forward pass matches Python | `node` |

`docs/` is what GitHub Pages serves, so the demo gets a live URL with no build
step.

The dependency arrows only ever point one way: `train` → `nn` → `engine`.
`engine.py` has no idea neural networks exist, and `nn.py` has no idea training
exists. That is what makes each piece testable on its own.

## Where training goes

In `train.py`, not in the library files. `engine.py` and `nn.py` define *what
things are*. `train.py` is the entry point that *does something with them*:
picks a dataset, builds a network, runs the loop. Swapping in a different
dataset or a different loss means editing one file and leaving the engine alone.

## Run it

```bash
python train.py
```

## The digit recognizer

`train_digits.py` trains a 64 → 16 → 10 network on 1,437 scanned handwritten
digits, using the same `Value` engine. `make_page.py` then bakes the learned
weights into `docs/index.html`, a single self-contained file: open it,
draw a digit with the mouse, and the network reads it. No server, no network
request, no dependencies.

```bash
python train_digits.py     # ~25 minutes; writes model.json and docs/index.html
open docs/index.html
```

It prints a live progress bar with an ETA while it runs, and rebuilds the demo
page automatically when it finishes.

### Why it needs `w_scale='auto'`

The first attempt got stuck around 50%. The cause was the weight
initialisation, not the training loop.

A neuron sums `nin` terms before its `tanh`, and the spread of that sum grows
with `sqrt(nin)`. With weights drawn from (-1, 1) — the video's choice, correct
for 3 inputs — a 64-input neuron starts far out on the flat tails of `tanh`,
where the local gradient `(1 - t²)` is nearly zero. The network begins the run
already saturated.

`MLP(64, [16, 10], w_scale='auto')` scales each layer's starting weights by
`1/sqrt(fan-in)`, which keeps the initial sum the same size regardless of layer
width. That is Xavier/Glorot initialisation. Test accuracy after one epoch went
from 32% to 61%.

The learning rate mattered just as much: 0.6 saturated the network outright and
pinned the loss at exactly 4.0. 0.08 works.

## Reading the output

```
step  12   loss   0.219546   sign  75.00%   close  81.34%
```

- **loss** — sum of squared errors, the thing being minimised
- **sign** — percent of predictions on the correct side of zero. With 4 examples
  this can only be 0, 25, 50, 75 or 100, so it moves in jumps and then sits still
- **close** — percent of the way from maximally wrong to exact, averaged over the
  examples. Moves a little every step, so this is the one to watch for progress

`close` scores each prediction as `1 - |error| / 2`, since a `tanh` output in
(-1, 1) against a target of ±1 can be off by at most 2.0.

## The training loop

Four steps per pass, always in this order:

1. **forward** — rebuild predictions and loss from the current weights
2. **zero grad** — reset every gradient to 0
3. **backward** — backpropagate
4. **update** — nudge each weight against its gradient

Both of the first two are easy to skip, and both fail quietly rather than
raising.

Skip the forward pass and the loss freezes. It is a number computed once, from a
graph built out of the weights as they were at that moment. Changing `p.data`
afterward does not reach back into it.

Skip the zeroing and gradients accumulate across passes, because every
`_backward` uses `+=`. Steps grow until a `tanh` saturates, and a saturated
`tanh` has a local gradient of roughly zero, so that neuron stops learning for
good. The symptom is predictions flipping sign and then refusing to move.
