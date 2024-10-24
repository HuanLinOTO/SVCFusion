import os
import gradio as gr
import torch
import yaml
from fap.utils.file import make_dirs
from SVCFusion.i18n import I
from SVCFusion.models.inited import (
    model_list,
)


class InstallModel:
    def old_install_model(self, file, model_name):
        package = torch.load(file, map_location="cpu")

        base_path = os.path.join("models", model_name)
        make_dirs(base_path)

        with open(os.path.join(base_path, "config.yaml"), "w") as f:
            f.write(yaml.dump(package["config"]))

        torch.save(package, os.path.join(base_path, "model.pt"))

    def install_model(self, file, model_name):
        if not file:
            gr.Info("请上传模型包或等待上传完成")
            return
        if not model_name:
            gr.Info("请填写模型名称")
            return
        gr.Info("模型安装中，请稍等")

        if file.name.endswith(".sf_pkg"):
            package = torch.load(file, map_location="cpu")
            model_type_index = package["model_type_index"]

            if hasattr(model_list[model_type_index], "install_model"):
                model_list[model_type_index].install_model(package, model_name)
            else:
                gr.Info("该模型不支持安装")
        elif file.name.endswith(".h0_ddsp_pkg_model"):
            self.old_install_model(file, model_name)
        else:
            gr.Info("不支持的模型包格式")
            return

        gr.Info("安装完成")

    def __init__(self) -> None:
        gr.Markdown(I.install_model.tip)

        self.file = gr.File(
            type="filepath",
            label=I.install_model.file_label,
        )

        self.model_name = gr.Textbox(
            label=I.install_model.model_name_label,
            placeholder=I.install_model.model_name_placeholder,
        )

        self.submit_btn = gr.Button(I.install_model.submit_btn_value, variant="primary")

        self.submit_btn.click(
            self.install_model,
            inputs=[self.file, self.model_name],
        )
