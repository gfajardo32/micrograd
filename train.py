"""Training: the dataset, the loss, and the gradient descent loop.

This is where training belongs. engine.py and nn.py are the library, they define
what a Value is and what a network is. They never train anything. This file is
the entry point: it picks a dataset, builds a network, and runs the loop.

Run it with:  python train.py

Follows Andrej Karpathy's lecture "The spelled-out intro to neural networks and
backpropagation: building micrograd" (https://www.youtube.com/watch?v=VMj-3S1tku0).
Original project: https://github.com/karpathy/micrograd (MIT). Learning exercise,
not original work.
"""

import random

from nn import MLP

# random.seed(1337)   # uncomment for runs that repeat exactly

# ---- dataset ------------------------------------------------------------

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]   # targets, one per input row


def mse_loss(ypred, ys):
    """Sum of squared errors. Zero when every prediction matches its target."""
    return sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))


def sign_accuracy(ypred, ys):
    """Percent of predictions on the correct side of zero.

    This is the classification-style read: did we get the sign right, never mind
    by how much. With 4 examples it can only ever be 0, 25, 50, 75 or 100, so it
    jumps in chunks and then sits still. Good for "is it right yet", useless for
    watching per-step progress.
    """
    correct = sum(1 for ygt, yout in zip(ys, ypred) if (yout.data >= 0) == (ygt >= 0))
    return 100.0 * correct / len(ys)


def closeness(ypred, ys):
    """Percent of the way from worst-possible to perfect, averaged over examples.

    tanh outputs live in (-1, 1) and targets are -1 or 1, so the worst any single
    prediction can be off is 2.0. Scoring each one as 1 - |error|/2 gives 100%
    for an exact hit and 0% for maximally wrong. Unlike sign accuracy this moves
    a little every step, which is what you want to watch during training.
    """
    total = sum(1.0 - abs(yout.data - ygt) / 2.0 for ygt, yout in zip(ys, ypred))
    return 100.0 * max(0.0, total / len(ys))


def train(model, xs, ys, steps=50, lr=0.05, verbose=True):
    """Four steps per pass, always in this order.

    1. forward    rebuild predictions and loss from the CURRENT weights
    2. zero grad  reset every gradient to 0
    3. backward   backpropagate
    4. update     nudge each weight against its gradient

    Skipping step 1 freezes the loss: it is a number computed once from a graph
    that no longer reflects the weights, so re-reading it shows the old value
    forever.

    Skipping step 2 lets gradients pile up across passes, because _backward uses
    +=. The steps grow until a tanh saturates, and a saturated tanh has a local
    gradient of nearly zero, so that neuron stops learning permanently. The
    symptom is predictions flipping sign and then refusing to move.
    """
    for k in range(steps):

        # 1. forward pass
        ypred = [model(x) for x in xs]
        loss = mse_loss(ypred, ys)

        # 2. zero the gradients
        model.zero_grad()

        # 3. backward pass
        loss.backward()

        # 4. update
        for p in model.parameters():
            p.data += -lr * p.grad

        if verbose:
            print(f"step {k:3d}   "
                  f"loss {loss.data:10.6f}   "
                  f"sign {sign_accuracy(ypred, ys):6.2f}%   "
                  f"close {closeness(ypred, ys):6.2f}%")

    return ypred, loss


if __name__ == "__main__":
    model = MLP(3, [4, 4, 1])
    print(model)
    print(len(model.parameters()), "parameters\n")

    ypred, loss = train(model, xs, ys, steps=50, lr=0.05)

    print("\npredictions:", [round(y.data, 4) for y in ypred])
    print("targets:    ", ys)
    print(f"sign accuracy: {sign_accuracy(ypred, ys):.2f}%")
    print(f"closeness:     {closeness(ypred, ys):.2f}%")
