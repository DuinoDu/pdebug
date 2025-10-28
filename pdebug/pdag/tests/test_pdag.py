"""Contents of hello_kedro.py""" ""
from dataclasses import dataclass

from pdebug.pdag.io import DataCatalog, MemoryDataSet
from pdebug.pdag.pipeline import node, pipeline
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


def backbone(feature: Feature):
    return Feature(data=f"backbone_{feature.data}")


def human_seg(feature: Feature, label: Label, meta_info: MetaInfo):
    return f"{feature.data} => human_seg!"


def mono_depth(feature: Feature, label: Label, meta_info: MetaInfo):
    return f"{feature.data} => mono_depth!"


def test_pdag():
    backbone_node = node(
        backbone, inputs="input_feature", outputs="backbone_feature"
    )
    human_seg_node = node(
        human_seg,
        inputs=["backbone_feature", "label", "meta_info"],
        outputs="human_seg_output",
    )
    mono_depth_node = node(
        mono_depth,
        inputs=["backbone_feature", "label", "meta_info"],
        outputs="mono_depth_output",
    )

    # Assemble nodes into a pipeline
    mtl_pipeline = pipeline([backbone_node, human_seg_node, mono_depth_node])

    # Create a runner to run the pipeline
    runner = SequentialRunner()

    # Run the pipeline, with given data
    data_catalog = DataCatalog(
        {
            "input_feature": MemoryDataSet(Feature()),
            "label": MemoryDataSet(Label()),
            "meta_info": MemoryDataSet(MetaInfo()),
        }
    )
    print(runner.run(mtl_pipeline, data_catalog))
