import os
from dataclasses import dataclass

from pdebug.pdag.frontend.simple_graph import (
    build_graph_from_topology,
    get_sub_graph,
    simplegraph_to_pdag,
    vis_graph,
)
from pdebug.pdag.io import DataCatalog, MemoryDataSet
from pdebug.pdag.runner import SequentialRunner


@dataclass
class Feature:
    data: str = "feature"


@dataclass
class Label:
    data: str = "label"


@dataclass
class MetaInfo:
    data: str = "meta_info"


class Backbone:
    def __call__(self, feature: Feature):
        return Feature(data=f"backbone_{feature.data}")


class HumanSeg:
    def __call__(self, feature: Feature, label: Label, meta_info: MetaInfo):
        return f"{feature.data} => human_seg!"


class MonoDepth:
    def __call__(self, feature: Feature, label: Label, meta_info: MetaInfo):
        return f"{feature.data} => mono_depth!"


def test_simple_graph(tmpdir):
    topology = [
        "backbone => human_seg, mono_depth",
    ]
    backbone = Backbone()
    human_seg = HumanSeg()
    mono_depth = MonoDepth()

    nodes_func = {
        "backbone": backbone,
        "human_seg": human_seg,
        "mono_depth": mono_depth,
    }
    nodes_inputs = {
        "backbone": ["input_feature"],
        "human_seg": ["label", "meta_info"],
        "mono_depth": ["label", "meta_info"],
    }
    graph = build_graph_from_topology(
        topology,
        nodes_func=nodes_func,
        nodes_inputs=nodes_inputs,
    )
    vis_graph(graph, os.path.join(tmpdir, "graph.png"))

    def forward(graph, data_catalog, task_or_tasks=None):
        if task_or_tasks:
            sub_graph = get_sub_graph(graph, task_or_tasks)
        else:
            sub_graph = graph
        graph_pipeline = simplegraph_to_pdag(sub_graph)
        runner = SequentialRunner()
        res = runner.run(graph_pipeline, data_catalog)
        return res

    data_catalog = DataCatalog(
        {
            "input_feature": MemoryDataSet(Feature()),
            "label": MemoryDataSet(Label()),
            "meta_info": MemoryDataSet(MetaInfo()),
        }
    )

    res = forward(graph, data_catalog)
    print(res)

    res = forward(graph, data_catalog, "human_seg")
    print(res)

    res = forward(graph, data_catalog, "mono_depth")
    print(res)

    res = forward(graph, data_catalog, ["mono_depth", "human_seg"])
    print(res)
