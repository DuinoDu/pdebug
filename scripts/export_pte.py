import torch
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from executorch.export import ExportRecipe, export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def get_model_and_input(name):
    if name == "vit_b":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vision_transformer.vit_b_16(weights=weights).eval()
        inputs = (torch.randn(1, 3, 224, 224), )
    elif name == "mobilenet_v2":
        model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
        inputs = (torch.randn(1, 3, 224, 224), )
    else:
        raise NotImplementedError
    return model, inputs

def export_common():
    model, inputs = get_model_and_input("vit_b")
    recipe = ExportRecipe()
    # ExportSession expects a list of example input tuples
    session = export(model, [inputs], recipe, name="vit_b")
    session.export()
    session.save_pte_file("vit_b.pte")
    print("vit_b.pte saved")

def export_xnnpack():
    model, inputs = get_model_and_input("vit_b")
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, inputs),
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    with open("vit_b.xnnpack.pte", "wb") as f:
        f.write(et_program.buffer)
    print("vit_b.xnnpack.pte saved")

def main():
    export_common()
    export_xnnpack()


if __name__ == "__main__":
    main()
