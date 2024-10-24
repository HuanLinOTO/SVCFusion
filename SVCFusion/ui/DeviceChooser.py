import gradio as gr
import torch

from SVCFusion.device import get_cuda_devices
from SVCFusion.i18n import I


class DeviceChooser:
    def refresh(self):
        self.cuda_devices = get_cuda_devices()
        return gr.update(
            choices=["CPU", *self.cuda_devices],
            value="CPU" if not torch.cuda.is_available() else self.cuda_devices[0],
        )

    def __init__(self, show=False, info=None) -> None:
        self.device_dropdown = gr.Dropdown(
            label=I.device_chooser.device_dropdown_label,
            value=self.refresh,
            info=info,
            type="index",
            interactive=True,
            visible=show,
        )

    def get_device_str_from_index(index):
        if not index:
            print("index is None", index)
            return "cpu"
        if index == 0:
            return "cpu"
        else:
            return f"cuda:{index - 1}"
