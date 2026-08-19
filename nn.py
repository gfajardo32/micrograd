"""Neural net building blocks, on top of the Value engine.

Neuron -> Layer -> MLP, each one a thin wrapper around the level below it.

Every level exposes parameters(), so a training loop can reach every weight and
bias in the whole network with a single call.

Follows Andrej Karpathy's lecture "The spelled-out intro to neural networks and
backpropagation: building micrograd" (https://www.youtube.com/watch?v=VMj-3S1tku0).
Original project: https://github.com/karpathy/micrograd (MIT). Learning exercise,
not original work.
"""

import random

from engine import Value


class Module:
    """Shared behaviour for anything that owns parameters."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, w_scale=1.0):
        """w_scale shrinks the starting weights.

        The default of 1.0 is what the video uses: weights drawn uniformly from
        (-1, 1). That is fine for 3 inputs. It is actively harmful for 64.

        A neuron sums nin terms before the tanh, so the spread of that sum grows
        with sqrt(nin). With 64 inputs and weights this size, most neurons start
        far out on the flat tails of tanh, where the local gradient (1 - t**2)
        is nearly zero. The network begins the run already saturated and barely
        learns. Passing w_scale=nin**-0.5 keeps the initial sum around the same
        size no matter how wide the layer is, which is the whole idea behind
        Xavier/Glorot initialisation.
        """
        self.w = [Value(random.uniform(-1, 1) * w_scale) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1) * w_scale)

    def __call__(self, x):
        # w . x + b, then squash into (-1, 1)
        # self.b is the starting value of the sum, so it is added exactly once
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, w_scale=1.0):
        # 'auto' means scale by 1/sqrt(fan-in), see Neuron for why
        s = nin ** -0.5 if w_scale == 'auto' else w_scale
        self.neurons = [Neuron(nin, s) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        # a single-neuron layer returns a bare Value, not a list of one
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts, w_scale=1.0):
        """nin: number of inputs. nouts: neuron count for each layer, in order.

        w_scale=1.0 reproduces the video. w_scale='auto' scales each layer's
        starting weights by 1/sqrt(fan-in), which matters once layers get wide.
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], w_scale) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
