import os
import shutil
import time
from traceback import print_exception

import torch
import yaml
from fap.utils.file import make_dirs
from package_utils.config import JSONReader, YAMLReader
from package_utils.i18n import I
from package_utils.ui.ModelChooser import ModelChooser
from package_utils.models.inited import (
    model_list,
    model_name_list,
)
import gradio as gr


class ModelManager:
    def pack(self):
        model_type_index = self.model_chooser.seleted_model_type_index
        result = self.model_chooser.selected_parameters
        if hasattr(model_list[model_type_index], "pack_model"):
            gr.Info(I.model_manager.packing_tip)
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
            gr.Info(I.model_manager.unpackable_tip)

    def clear_log(self):
        search_path = self.model_chooser.selected_search_path
        model_type_index = self.model_chooser.seleted_model_type_index
        log_path = os.path.join(search_path, "logs")
        if os.path.exists(log_path):
            shutil.rmtree(log_path)

        # sovits only
        if model_type_index == 2:
            # 删除 search path 下面的 所有 D_ 开头的 .pth 文件
            for file in os.listdir(search_path):
                if file.startswith("D_") and file.endswith(".pth"):
                    os.remove(os.path.join(search_path, file))

    def change_model_type(self, model_type):
        model_type_index = model_name_list.index(model_type)
        search_path = self.model_chooser.selected_search_path
        try:
            if os.path.exists(os.path.join(search_path, "config.json")):
                with JSONReader(os.path.join(search_path, "config.json")) as f:
                    f["model_type_index"] = model_type_index
                with open(os.path.join(search_path, "config.json"), "w") as f:
                    f.write(f.json())
            elif os.path.exists(os.path.join(search_path, "config.yaml")):
                with YAMLReader(os.path.join(search_path, "config.yaml")) as config:
                    config["model_type_index"] = model_type_index
                with open(os.path.join(search_path, "config.yaml"), "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
            gr.Info(I.model_manager.change_success_tip)

        except Exception as e:
            gr.Info(I.model_manager.change_fail_tip)
            print_exception(e)

    def __init__(self) -> None:
        self.model_chooser = ModelChooser(
            show_options=False,
            show_submit_button=False,
        )
        self.pack_btn = gr.Button(
            I.model_manager.pack_btn_value,
            variant="primary",
        )
        self.output_file = gr.File(
            type="filepath",
            label=I.model_manager.pack_result_label,
            visible=False,
        )
        self.clear_log_btn = gr.Button(
            I.model_manager.clean_log_btn_value,
            variant="primary",
        )

        gr.Markdown(I.model_manager.change_model_type_info)
        self.model_type_dropdown = gr.Dropdown(
            label=I.model_chooser.model_type_dropdown_label,
            choices=model_name_list,
            value=model_name_list[0],
            interactive=True,
        )
        self.change_model_type_btn = gr.Button(
            I.model_manager.change_model_type_btn_value,
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

        self.change_model_type_btn.click(
            self.change_model_type,
            inputs=[
                self.model_type_dropdown,
            ],
        )
