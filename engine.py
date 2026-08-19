"""Scalar autograd engine.

A Value wraps a single number and remembers how it was produced, so we can walk
that history backwards and compute derivatives.

Every operation does three things:
  1. compute the forward value
  2. record its children and the op symbol, so the graph can be traced
  3. define a _backward closure that pushes gradient from the output to the inputs

Gradients accumulate with += rather than =. A node used twice (like a + a)
receives a contribution along each path, and overwriting would discard one.

Follows Andrej Karpathy's lecture "The spelled-out intro to neural networks and
backpropagation: building micrograd" (https://www.youtube.com/watch?v=VMj-3S1tku0).
Original project: https://github.com/karpathy/micrograd (MIT). Learning exercise,
not original work.
"""

import math


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ---- the three primitive operations --------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # ---- everything else is built on those three -----------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    # reversed forms, for when a plain number is on the left:
    # 2 * v, and sum([...]) which starts its accumulator at the integer 0
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- activations ---------------------------------------------------

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # ---- backpropagation -----------------------------------------------

    def backward(self):
        """Run backprop from this node through the whole graph behind it."""
        # topological order: every node appears after all of its children
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
