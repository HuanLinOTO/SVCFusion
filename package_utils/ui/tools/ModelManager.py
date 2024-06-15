from cProfile import label
import os
import shutil
import time

import torch
from fap.utils.file import make_dirs
from package_utils.ui.ModelChooser import ModelChooser
from package_utils.models.inited import (
    model_list,
)
import gradio as gr


class ModelManager:
    def pack(self):
        model_type_index = self.model_chooser.seleted_model_type_index
        result = self.model_chooser.selected_parameters
        if hasattr(model_list[model_type_index], "pack_model"):
            gr.Info("正在打包，请勿多次点击")
            packed_model = model_list[model_type_index].pack_model(result)
            packed_model["model_type_index"] = model_type_index
            make_dirs("tmp/packed_models")
            # yymmdd_HHMMSS
            name = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
            output_path = f"tmp/packed_models/{name}.{model_list[model_type_index].model_name}.sf_pkg"
            torch.save(packed_model, output_path)
            return gr.update(
                value=output_path,
                visible=True,
            )
        else:
            gr.Info("该模型不支持打包")

    def clear_log(self):
        search_path = self.model_chooser.selected_search_path
        model_type_index = self.model_chooser.seleted_model_type_index
        shutil.rmtree(os.path.join(search_path, "logs"))

        # sovits only
        if model_type_index == 2:
            # 删除 search path 下面的 所有 D_ 开头的 .pth 文件
            for file in os.listdir(search_path):
                if file.startswith("D_") and file.endswith(".pth"):
                    os.remove(os.path.join(search_path, file))

    def __init__(self) -> None:
        self.model_chooser = ModelChooser(
            show_options=False,
            show_submit_button=False,
        )
        self.pack_btn = gr.Button(
            "打包模型",
            variant="primary",
        )
        self.output_file = gr.File(
            type="filepath",
            label="打包结果",
            visible=False,
        )
        self.clear_log_btn = gr.Button(
            "清空日志(确认不再训练再清空)",
            variant="primary",
        )

        self.pack_btn.click(
            self.pack,
            outputs=[
                self.output_file,
            ],
        )

        self.clear_log_btn.click(
            self.clear_log,
        )
