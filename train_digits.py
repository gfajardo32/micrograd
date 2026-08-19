"""Train a handwritten-digit classifier using the scalar engine in engine.py.

Data: 1797 scanned 8x8 digits from the UCI "Optical Recognition of Handwritten
Digits" set, bundled here as digits.csv so this script needs nothing but the
standard library plus engine.py and nn.py.

Network: 64 inputs (one per pixel) -> 16 hidden -> 10 outputs, one per digit.
Every neuron is a tanh, so each output sits in (-1, 1). The target for an image
of a 7 is +1 on output 7 and -1 on the other nine. To read a prediction, take
whichever output is largest.

This is slow. The engine builds a Python object per arithmetic operation, so a
single image costs about 1200 of them on the forward pass and as many again
going back. That is the price of being able to see every step.

Run:  python train_digits.py
Writes: model.json
"""

import csv
import json
import math
import random
import sys
import time

from nn import MLP

SEED = 1337
HIDDEN = 16
EPOCHS = 20
BATCH = 32
LR = 0.08
TEST_FRACTION = 0.2


def load_digits(path="digits.csv"):
    """Return (X, y). Pixels are 0-16 in the file, rescaled here to 0-1."""
    X, y = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            X.append([int(row[f"p{i}"]) / 16.0 for i in range(64)])
            y.append(int(row["label"]))
    return X, y


def split(X, y, test_fraction, seed):
    """Shuffle once, then cut. The test rows are never trained on."""
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_fraction))
    tr, te = idx[:cut], idx[cut:]
    return ([X[i] for i in tr], [y[i] for i in tr],
            [X[i] for i in te], [y[i] for i in te])


def target_vector(label):
    """One-vs-all: +1 on the right output, -1 everywhere else."""
    return [1.0 if i == label else -1.0 for i in range(10)]


def predict(model, x):
    """Index of the largest output. Uses the engine, so it is slow."""
    outs = model(x)
    return max(range(10), key=lambda i: outs[i].data)


def accuracy(model, X, y):
    correct = sum(1 for xi, yi in zip(X, y) if predict(model, xi) == yi)
    return 100.0 * correct / len(y)


def export(model, path="model.json"):
    """Dump the learned weights as plain numbers, for the browser to reuse."""
    layers = []
    for layer in model.layers:
        layers.append({
            "w": [[w.data for w in neuron.w] for neuron in layer.neurons],
            "b": [neuron.b.data for neuron in layer.neurons],
        })
    with open(path, "w") as f:
        json.dump({"activation": "tanh", "layers": layers}, f)
    return path


def clock(seconds):
    """Seconds as 4m 07s, or 1h 02m once it gets long."""
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def progress(done, total, started, note=""):
    """One self-overwriting line: bar, percentage, elapsed and estimated remaining."""
    frac = done / total
    width = 24
    filled = int(width * frac)
    bar = "#" * filled + "." * (width - filled)
    elapsed = time.time() - started
    # only guess at a finish time once there is enough history to be worth trusting
    eta = f"eta {clock(elapsed / frac - elapsed)}" if frac > 0.02 else "eta ..."
    line = f"\r  [{bar}] {frac*100:5.1f}%   {clock(elapsed)} elapsed   {eta}   {note}"
    sys.stdout.write(line.ljust(96))
    sys.stdout.flush()


def train():
    random.seed(SEED)

    X, y = load_digits()
    Xtr, ytr, Xte, yte = split(X, y, TEST_FRACTION, SEED)
    print(f"{len(Xtr)} training images, {len(Xte)} held out for testing")

    # w_scale='auto' is load-bearing here, see the note in nn.Neuron
    model = MLP(64, [HIDDEN, 10], w_scale='auto')
    print(f"{len(model.parameters())} parameters")
    print(f"{EPOCHS} epochs, batch size {BATCH}, learning rate {LR}")
    print("this is a scalar engine in pure Python, so expect roughly 25 minutes\n")

    order = list(range(len(Xtr)))
    n_batches = (len(order) + BATCH - 1) // BATCH
    total_steps = EPOCHS * n_batches
    started = time.time()

    for epoch in range(EPOCHS):
        lr = LR
        random.shuffle(order)

        running = 0.0
        batches = 0

        for start in range(0, len(order), BATCH):
            chunk = order[start:start + BATCH]

            progress(epoch * n_batches + batches, total_steps, started,
                     f"epoch {epoch+1}/{EPOCHS}")

            # 1. forward pass over the mini-batch
            total = 0
            for i in chunk:
                outs = model(Xtr[i])
                tgt = target_vector(ytr[i])
                total = total + sum((o - t)**2 for o, t in zip(outs, tgt))
            loss = total * (1.0 / len(chunk))

            # 2. zero the gradients
            model.zero_grad()

            # 3. backward pass
            loss.backward()

            # 4. update
            for p in model.parameters():
                p.data += -lr * p.grad

            running += loss.data
            batches += 1

        train_loss = running / batches
        # scoring every held-out image each epoch costs more than the epoch
        # itself, so sample during training and do the full set at the end
        test_acc = accuracy(model, Xte[:150], yte[:150])

        sys.stdout.write("\r" + " " * 96 + "\r")
        print(f"epoch {epoch+1:2d}/{EPOCHS}   "
              f"loss {train_loss:7.4f}   test acc {test_acc:6.2f}%   "
              f"{clock(time.time() - started)} elapsed")

    progress(total_steps, total_steps, started, "done")
    sys.stdout.write("\r" + " " * 96 + "\r")

    print(f"\ntrained in {clock(time.time() - started)}")
    print("scoring every image, this takes a few seconds...")
    train_acc = accuracy(model, Xtr, ytr)
    test_acc = accuracy(model, Xte, yte)
    print(f"  train accuracy: {train_acc:.2f}%   ({len(Xtr)} images it learned from)")
    print(f"  test accuracy:  {test_acc:.2f}%   ({len(Xte)} images it never saw)")
    print(f"\nwrote {export(model)}")

    # build the browser page straight away, so one command gives a finished thing
    try:
        import make_page
        page = make_page.build(test_acc)
        print(f"wrote {page}   <- open this one")
    except Exception as exc:
        print(f"could not build the page ({exc}); run: python make_page.py {test_acc:.2f}")

    return model


if __name__ == "__main__":
    train()
