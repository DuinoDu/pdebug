"""Simple graph.

Simple graph is a dict, with below data structure:
    {
        "node_name_1": Node,
        "node_name_2": Node,
        "node_name_3": Node,
    }

"""
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from pdebug.pdag.pipeline import node as _Node
from pdebug.pdag.pipeline import pipeline as Pipeline

__all__ = [
    "build_graph_from_topology",
    "vis_graph",
    "simplegraph_to_pdag",
    "build_from_topology",
]


@dataclass
class Node:
    upstream: List[str] = ()
    downstream: List[str] = ()
    inputs: List[str] = ()
    outputs: str = None
    func: Callable = None

    def __post_init__(self):
        self.upstream = list(self.upstream)
        self.downstream = list(self.downstream)
        self.inputs = list(self.inputs)
        if self.func is None:
            self.func = lambda x: x

    def __iadd__(self, item: "Node"):
        self.upstream.extend(item.upstream)
        self.downstream.extend(item.downstream)
        self.inputs.extend(item.inputs)
        self.outputs = item.outputs
        self.unique()
        return self

    def unique(self) -> None:
        unique_fn = lambda x: sorted(set(x), key=x.index)
        self.upstream = unique_fn(self.upstream)
        self.downstream = unique_fn(self.downstream)
        self.inputs = unique_fn(self.inputs)

    def __repr__(self) -> str:
        _str = f"  upstream: {self.upstream}"
        _str += f"\n  downstream: {self.downstream}"
        _str += f"\n  inputs: {self.inputs}"
        _str += f"\n  outputs: {self.outputs}"
        func_str = self.func.__str__()[:30].replace("\n", " ") + " ..."
        _str += f"\n  func: {func_str}"
        return _str


def build_graph_from_topology(
    topology: List[str],
    nodes_func: Optional[Dict[str, Callable]] = None,
    nodes_inputs: Optional[Dict[str, str]] = None,
    link_symbol: str = "=>",
) -> Dict:
    """
    Build graph from topology.

    Args:
        topology: input topology list.
        nodes_func: each node's forward function.
        nodes_inputs: each node's extra input name.
        link_symbol: topology link symbol, default is '=>'
    """
    # TODO: check topology.
    assert isinstance(topology, list)
    assert len(topology) >= 1, "topology can't be empty!"

    # convert multi "=>" one line to multiple lines.
    short_topology = []
    for link in topology:
        assert link_symbol in link
        nodes = [n.strip() for n in link.split(link_symbol)]
        assert len(nodes) >= 2
        if len(nodes) == 2:
            short_topology.append(link)
            continue
        for prev, cur in zip(nodes[:-1], nodes[1:]):
            short_topology.append(f"{prev}{link_symbol}{cur}")
    topology = short_topology

    # build graph based on topology.

    def _process_comma(node):
        if "," in node:
            node = [n.strip() for n in node.split(",") if n.strip() != ""]
        else:
            node = [node]
        return node

    def _add_node_pair(graph, left, right):
        assert isinstance(left, str)
        assert isinstance(right, str)
        if left not in graph:
            graph[left] = Node()
        if right not in graph:
            graph[right] = Node()

        # set link
        graph[left].downstream.append(right)
        graph[right].upstream.append(left)

        # set node func
        if nodes_func:
            graph[left].func = nodes_func.get(left, None)
            graph[right].func = nodes_func.get(right, None)
        # append extra node inputs
        if nodes_inputs:
            graph[left].inputs.extend(nodes_inputs.get(left, []))
            graph[right].inputs.extend(nodes_inputs.get(right, []))

        graph[left].unique()
        graph[right].unique()

    graph = dict()
    for link in topology:
        left, right = [n.strip() for n in link.split(link_symbol)]
        left = _process_comma(left)
        right = _process_comma(right)

        assert not (
            len(left) > 1 and len(right) > 1
        ), "Not support multiple nodes both for left and right node."
        if len(left) > 1:
            assert len(right) == 1
            for each_left in left:
                _add_node_pair(graph, each_left, right[0])
        elif len(right) > 1:
            assert len(left) == 1
            for each_right in right:
                _add_node_pair(graph, left[0], each_right)
        else:
            _add_node_pair(graph, left[0], right[0])
    return graph


def vis_graph(graph: Dict, output: str = "graph.png"):
    """Visualize graph using graphviz."""
    try:
        from graphviz import Digraph
    except ModuleNotFoundError as e:
        print("Please install graphviz to vis graph.")
        return

    def _node_exists(g, node):
        nodes = [n for n in g.body if "->" not in n]
        return sum([node in n for n in nodes]) > 0

    def _edge_exists(g, left, right):
        edges = [e for e in g.body if "->" in e]
        return sum([left in e and right in e for e in edges]) > 0

    g = Digraph("GraphModule")
    for name in graph:
        if not _node_exists(g, name):
            g.node(name=name, color="red")
        for upstream in graph[name].upstream:
            if not _edge_exists(g, upstream, name):
                g.edge(upstream, name, color="green")
        for downstream in graph[name].downstream:
            if not _edge_exists(g, name, downstream):
                g.edge(name, downstream, color="green")

    g.render(output, view=False)


def clear_graph(graph: Dict):
    """Clear graph to ensure all nodes are in graph.keys()."""
    for name in graph:
        valid = [n for n in graph[name].upstream if n in graph]
        graph[name].upstream = valid
        valid = [n for n in graph[name].downstream if n in graph]
        graph[name].downstream = valid


def merge_graph(g1: Dict, g2: Dict) -> Dict:
    """Merge two graph."""
    out = copy.deepcopy(g1)
    for node in g2:
        if node in out:
            out[node] += g2[node]
        else:
            out[node] = g2[node]
    return out


def is_input_node(node: Node):
    """If is input node."""
    return len(node.upstream) == 0


def is_output_node(node: Node):
    """If is output node."""
    return len(node.downstream) == 0


def get_input_nodes(graph: Dict):
    """Get input node in graph."""
    return [name for name in graph if is_input_node(graph[name])]


def get_output_nodes(graph: Dict):
    """Get input node in graph."""
    return [name for name in graph if is_output_node(graph[name])]


def get_sub_graph(
    graph: Dict,
    output_node: Union[str, List[str]],
    input_node: Optional[str] = None,
) -> Dict:
    """Get subgraph by output_node, using dfs."""
    graph = copy.deepcopy(graph)
    sub_graph = dict()
    if isinstance(output_node, list):
        for each_output_node in output_node:
            cur_sub_graph = get_sub_graph(graph, each_output_node)
            sub_graph = merge_graph(sub_graph, cur_sub_graph)
        clear_graph(sub_graph)
        return sub_graph

    stack = [output_node]
    visited = {output_node}
    sub_graph[output_node] = graph[output_node]

    while True:
        if len(stack) > 0:
            node = stack.pop()
            for i in graph[node].upstream:
                if i not in visited:
                    stack.append(i)
                    visited.add(i)
                sub_graph[i] = graph[i]
        else:
            break

    clear_graph(sub_graph)
    return sub_graph


def simplegraph_to_pdag(graph: Dict) -> Pipeline:
    """Convert simplegraph to pdag's pipeline."""

    graph = copy.deepcopy(graph)
    # add link info in node inputs and outputs
    for node in graph:
        output_name = f"{node}.output"
        graph[node].outputs = output_name  # .append(output_name)
        for down_node in graph[node].downstream:
            graph[down_node].inputs.insert(0, output_name)
            # graph[down_node].inputs.append(output_name)

    # pdag interface
    all_dag_nodes = []
    for name, node in graph.items():
        dag_node = _Node(node.func, inputs=node.inputs, outputs=node.outputs)
        all_dag_nodes.append(dag_node)

    return Pipeline(all_dag_nodes)


def build_from_topology(
    topology: List[str],
    nodes_func: Optional[Dict[str, Callable]] = None,
    nodes_inputs: Optional[Dict[str, str]] = None,
    link_symbol: str = "=>",
) -> Pipeline:
    graph = build_graph_from_topology(
        topology, nodes_func, nodes_inputs, link_symbol
    )
    pipeline_ = simplegraph_to_pdag(graph)
    return pipeline_
