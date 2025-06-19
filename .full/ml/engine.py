from graphviz import Digraph
import numpy as np


class Scalar:
    def __init__(self, data, _children=(), _op="", label=None):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        if label is not (None):
            self.label = label
        else:
            self.label = f"Scalar({data:.4g})"

    def __repr__(self):
        return f"Scalar({self.data}, grad = {self.grad})"

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        output = Scalar(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        output = Scalar(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __neg__(self):
        if not isinstance(self, Scalar):
            self = Scalar(self)
        output = Scalar(-self.data, _children=(self,), _op="neg")

        def _backward():
            self.grad -= output.grad

        output._backward = _backward
        return output

    def __sub__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        output = self + (-other)
        return output

    def __pow__(self, power):
        if not isinstance(power, Scalar):
            power = Scalar(power)
        output = Scalar(self.data**power.data, _children=(self, power), _op="**")

        def _backward():
            self.grad += (power.data * (self.data ** (power.data - 1))) * output.grad

        output._backward = _backward
        return output

    def ifel(self, condition, true_case, false_case):
        if not isinstance(true_case, Scalar):
            true_case = Scalar(true_case)
        if not isinstance(false_case, Scalar):
            false_case = Scalar(false_case)

        output = Scalar(
            true_case.data if condition else false_case.data,
            _children=(self,),
            _op="ifel",
        )

        def _backward():
            if condition:
                true_case.grad += output.grad
            else:
                false_case.grad += output.grad
            self.grad += output.grad

        output._backward = _backward
        return output

    def __gt__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        output = Scalar(self.data > other.data, _children=(self, other), _op=">")

        def _backward():
            if output.data:
                self.grad += output.grad
            else:
                other.grad += output.grad

        output._backward = _backward
        return output

    def __lt__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        output = Scalar(self.data < other.data, _children=(self, other), _op="<")

        def _backward():
            if output.data:
                self.grad += output.grad
            else:
                other.grad += output.grad

        output._backward = _backward
        return output

    def tanh(self):
        output = Scalar(np.tanh(self.data), _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - output.data * output.data) * output.grad

        output._backward = _backward
        return output

    def backward(self):
        # Topological order all of the children in the graph
        topology = []
        visited = set()

        def build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topology(child)
                topology.append(v)

        build_topology(self)

        # Iterate one variable at a time
        self.grad = 1
        for v in reversed(topology):
            v._backward()

    def trace(self):
        """
        Trace the computation graph of the Scalar object and return nodes and edges.
        """
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(self)
        return nodes, edges

    def visualize_graph(self, output_label=None):
        """
        Visualize the computation graph of the Scalar object using Graphviz.
        """
        nodes, edges = self.trace()
        graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        for n in nodes:
            if output_label is not None:
                graph.node(
                    name=str(id(n)),
                    label=f"{n.label} | data: {n.data:.4f} | D[{output_label}, {n.label}]: {n.grad:.4f}",
                    shape="record",
                    style="filled",
                    fillcolor="white",
                )
            else:
                graph.node(
                    name=str(id(n)),
                    label=f"{n.label} | data: {n.data:.4f} | grad: {n.grad:.4f}",
                    shape="record",
                    style="filled",
                    fillcolor="white",
                )
            if n._op:
                graph.node(
                    name=str(id(n)) + n._op,
                    label=n._op,
                    shape="circle",
                    facecolor="lightgray",
                    style="filled",
                )
                graph.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            graph.edge(str(id(n1)), str(id(n2)) + n2._op)

        return graph
