import os
import shutil
from traceback import print_exception

import torch
import yaml
from fap.utils.file import make_dirs
from SVCFusion.config import JSONReader, YAMLReader
from SVCFusion.const_vars import WORK_DIR_PATH
from SVCFusion.dataset_utils import get_spk_from_dir
from SVCFusion.i18n import I
from SVCFusion.ui.ModelChooser import ModelChooser
from SVCFusion.models.inited import (
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
            # name = time.strftime("%y%m%d_%H-%M-%S", time.localtime())
            output_path = f"tmp/packed_models/{self.get_dst_name()}.sf_pkg"
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

    def move_folder(self, dst_name):
        gr.Info(I.model_manager.moving_tip)
        search_path = self.model_chooser.selected_search_path

        shutil.move(search_path, "models/" + dst_name)

        if search_path == WORK_DIR_PATH:
            os.makedirs(search_path)

        gr.Info(I.model_manager.moved_tip.replace("{1}", "models/" + dst_name))

    def get_dst_name(self):
        search_path = self.model_chooser.selected_search_path

        spks = get_spk_from_dir(search_path)
        model_type = model_name_list[self.model_chooser.seleted_model_type_index]

        result: str = model_type

        if len(spks) == 1:
            result += f"_{spks[0]}"
            return result.replace(" ", "_")

        for spk in spks:
            tmp = result + f"_{spk}"

            if len(tmp + I.model_manager.other_text) > 10:
                result = tmp + I.model_manager.other_text
                break
            else:
                result = tmp

        return result.replace(" ", "_")

    def __init__(self) -> None:
        gr.Markdown("## " + I.model_manager.choose_model_title)
        self.model_chooser = ModelChooser(
            show_options=False,
            show_submit_button=False,
        )
        gr.Markdown("## " + I.model_manager.action_title)
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
            variant="stop",
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
        gr.HTML('<div style="height: 20px;"></div>')
        gr.Markdown(I.model_manager.move_folder_tip)
        self.move_folder_dst_name = gr.Textbox(
            label=I.model_manager.move_folder_name,
        )
        self.move_folder_dst_name_auto_get_btn = gr.Button(
            I.model_manager.move_folder_name_auto_get,
        )
        self.move_folder_btn = gr.Button(
            I.model_manager.move_folder_btn_value,
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

        self.move_folder_dst_name_auto_get_btn.click(
            self.get_dst_name, outputs=[self.move_folder_dst_name]
        )

        self.move_folder_btn.click(
            self.move_folder,
            inputs=[self.move_folder_dst_name],
        )
