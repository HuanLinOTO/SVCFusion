import gradio as gr
import torch
import cpuinfo

from package_utils.device import get_cuda_devices
from package_utils.i18n import I


class DeviceChooser:
    def refresh(self):
        self.cuda_devices = get_cuda_devices()
        cpu = cpuinfo.get_cpu_info()["brand_raw"]
        return gr.update(
            choices=[f"CPU: {cpu}", *self.cuda_devices],
            value=f"CPU: {cpu}"
            if not torch.cuda.is_available()
            else self.cuda_devices[0],
        )

    def __init__(self, show=False) -> None:
        self.device_dropdown = gr.Dropdown(
            label=I.device_chooser.device_dropdown_label,
            value=self.refresh,
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
