import time

import torch
from fap.utils.file import make_dirs
from package_utils.ui.ModelChooser import ModelChooser
from package_utils.models.inited import (
    model_list,
)
import gradio as gr


class ModelManager:
    def pack(self, model_type_index, result):
        if hasattr(model_list[model_type_index], "pack_model"):
            packed_model = model_list[model_type_index].pack_model(result)
            make_dirs("tmp/packed_models")
            # hhmmss
            name = time.strftime("%H.%M.%S", time.localtime())
            output_path = f"tmp/packed_models/{name}.{model_list[model_type_index].model_name}.sf_pkg"
            torch.save(packed_model, output_path)

        else:
            gr.Info("该模型不支持打包")

    def __init__(self) -> None:
        self.model_chooser = ModelChooser(
            show_options=False,
            on_submit=self.pack,
        )
