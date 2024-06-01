import gradio as gr
import torch

from package_utils.device import get_cuda_devices


class DeviceChooser:
    def refresh(self):
        self.cuda_devices = get_cuda_devices()
        return gr.update(
            choices=["cpu", *self.cuda_devices],
            value="cpu" if not torch.cuda.is_available() else self.cuda_devices[0],
        )

    def __init__(self) -> None:
        self.device_dropdown = gr.Dropdown(
            label="设备",
            value=self.refresh,
            type="index",
            interactive=True,
        )

    def get_device_str_from_index(index):
        if index == 0:
            return "cpu"
        else:
            return f"cuda:{index - 1}"
