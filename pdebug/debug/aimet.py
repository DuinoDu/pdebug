import json

__all__ = ["print_quant_sim", "print_encoding_file"]


def print_quant_sim(
    quant_sim, name="model_layers_0_self_attn_v_proj_sha_0_Conv"
):
    model = quant_sim.model
    for module_name, module in model.named_modules():
        if name not in module_name:
            continue
        if not hasattr(module, "output_quantizers"):
            continue
        print(
            "quant_sim: ",
            name,
            module.output_quantizers[0],
            str(module.output_quantizers[0].use_symmetric_encodings),
        )


def print_encoding_file(
    encoding_file, name="model_layers_0_self_attn_v_proj_sha_0_Conv"
):
    activation_encoding_dict = json.load(open(encoding_file, "r"))[
        "activation_encodings"
    ]
    assert name in activation_encoding_dict
    output_encoding = activation_encoding_dict[module_name]["output"]
    print("encoding_file:", name, output_encoding["0"]["is_symmetric"])

    """
        module_name                                     encoding_dict   self.quant_sim.model
        model_layers_0_self_attn_v_proj_sha_0_Conv      True            False
        model_layers_0_self_attn_v_proj_sha_1_Conv      True            False
        ...
        model_layers_0_self_attn_v_proj_sha_31_Conv     True            False
        ...
        model_layers_0_self_attn_Transpose_128          True            False
        model_layers_0_self_attn_Concat_64              False           True

    """
