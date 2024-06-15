import gradio as gr
import torch
from package_utils.models.inited import (
    model_list,
)


class InstallModel:
    def install_model(self, file, model_name):
        if not file:
            gr.Info("请上传模型包或等待上传完成")
            return
        if not model_name:
            gr.Info("请填写模型名称")
            return
        gr.Info("模型安装中，请稍等")
        package = torch.load(file, map_location="cpu")
        model_type_index = package["model_type_index"]
        if hasattr(model_list[model_type_index], "install_model"):
            model_list[model_type_index].install_model(package, model_name)
        else:
            gr.Info("该模型不支持打包")

        gr.Info("安装完成")

    def __init__(self) -> None:
        gr.Markdown("""
        ## 目前仅支持上传 .sf_pkg 格式的新模型包

        旧模型包请转换后再上传
        # """)

        self.file = gr.File(
            type="filepath",
            label="上传模型包",
        )

        self.model_name = gr.Textbox(
            label="模型名称",
            placeholder="请输入模型名称",
        )

        self.submit_btn = gr.Button("安装模型", variant="primary")

        self.submit_btn.click(
            self.install_model,
            inputs=[self.file, self.model_name],
        )
