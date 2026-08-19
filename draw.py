"""Graphviz rendering of a Value's computation graph.

Optional. Needs the graphviz Python package and the graphviz binaries.
Mostly useful from a notebook, where the returned object renders inline:

    from engine import Value
    from draw import draw_dot

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b; c.label = 'c'
    c.backward()
    draw_dot(c)

Follows Andrej Karpathy's lecture "The spelled-out intro to neural networks and
backpropagation: building micrograd" (https://www.youtube.com/watch?v=VMj-3S1tku0).
Original project: https://github.com/karpathy/micrograd (MIT). Learning exercise,
not original work.
"""

from graphviz import Digraph


def trace(root):
    """Collect every node and edge reachable from root."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # every Value becomes a rectangular record node
        dot.node(name=uid,
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                 shape='record')
        if n._op:
            # a Value produced by an op also gets a small op node feeding into it
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
