import os
import tempfile
from collections import defaultdict

import numpy as np

from .env import TORCH_INSTALLED

if TORCH_INSTALLED:
    import torch

__all__ = ["QnnModel"]


class QnnModel:
    """
    Inference model using qnn_net_run.

    Example:
        >>> from pdebug.utils.qualcomm_utils import QnnModel
        >>> model = QnnModel(None, "...model.so", target="host")
        >>> model.set_output_info(["reg", "cls"], [(1,1,28), (1,1,1)])
        >>> y = model(x)
    """

    def __init__(
        self,
        context_binary,
        compiled_binary,
        target="host",
        convert_chw_to_hwc=True,
        device_id: str = None,  # MUST when target is android
    ):
        QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")
        assert QNN_SDK_ROOT
        target_list = ["host", "android"]
        assert target in target_list
        target_names = ["x86_64-linux-clang", "aarch64-android"]
        self.target = target_names[target_list.index(target)]
        self.context_binary = context_binary
        self.compiled_binary = compiled_binary
        self.convert_chw_to_hwc = convert_chw_to_hwc
        self.device_id = device_id
        if target == "android":
            assert device_id, "device_id is required when target is android"

        # self.output_names = ["conv2d_21", "conv2d_31"]
        # self.output_shapes = [(1, 1, 1404), (1, 1, 1)]
        self._output_names = None
        self._output_shapes = None

    def set_output_info(self, output_names, output_shapes):
        assert len(output_names) == len(output_shapes)
        self._output_names = output_names
        self._output_shapes = output_shapes

    @property
    def output_names(self):
        if not self._output_names:
            raise RuntimeError(
                "Forget to call `set_output_info` for QnnModel?"
            )
        return self._output_names

    @property
    def output_shapes(self):
        if not self._output_shapes:
            raise RuntimeError("Forget to set `set_output_info` for QnnModel?")
        return self._output_shapes

    def __call__(self, x):
        # assert x.ndim == 4

        with tempfile.TemporaryDirectory() as temp_calib_dir:
            list_txt = os.path.join(temp_calib_dir, "input_list.txt")
            list_fid = open(list_txt, "w")
            cnt = 0
            for x_i in x:
                raw_filename = os.path.join(temp_calib_dir, f"{cnt:06d}.raw")
                if self.convert_chw_to_hwc:
                    x_i = x_i.permute(1, 2, 0)
                x_i.cpu().numpy().tofile(raw_filename)
                list_fid.write(raw_filename + "\n")
                cnt += 1
            list_fid.close()

            output_dir = os.path.join(temp_calib_dir, "output_dir")
            os.makedirs(output_dir, exist_ok=True)
            if "android" in self.target:
                raise NotImplementedError
            else:
                self.qnn_net_run(list_txt, output_dir)

            preds = defaultdict(list)
            for cnt in range(x.shape[0]):
                for output_name, output_shape in zip(
                    self.output_names, self.output_shapes
                ):
                    output_raw_file = os.path.join(
                        output_dir, f"Result_{cnt}/{output_name}.raw"
                    )
                    output_data = np.fromfile(
                        output_raw_file, dtype=np.float32
                    ).reshape(output_shape)
                    preds[output_name].append(output_data)
            for name in preds:
                preds[name] = torch.Tensor(np.asarray(preds[name])).to(
                    x.device
                )
            return preds

    def qnn_net_run(self, input_list, output_dir):
        QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")
        if self.context_binary:
            cmd = f"""\
            {QNN_SDK_ROOT}/bin/{self.target}/qnn-net-run \
                --backend {QNN_SDK_ROOT}/lib/{self.target}/libQnnHtp.so \
                --retrieve_context {self.context_binary} \
                --input_list {input_list} \
                --output_dir {output_dir}
            """
        elif self.compiled_binary:
            cmd = f"""\
            {QNN_SDK_ROOT}/bin/{self.target}/qnn-net-run \
                --backend {QNN_SDK_ROOT}/lib/{self.target}/libQnnHtp.so \
                --model {self.compiled_binary} \
                --input_list {input_list} \
                --output_dir {output_dir}
            """
        os.system(cmd)
