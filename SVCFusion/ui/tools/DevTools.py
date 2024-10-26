from os import system
import gradio as gr

from SVCFusion.dlc import MetaV1, pack_directory_to_dlc_file


class DevTools:
    def pack_pretrain_dlc(self, directory_path, model_name):
        meta: MetaV1 = {
            "version": "v1",
            "type": "pretrain",
            "attrs": {
                "model": model_name,
            },
        }
        print("packing")
        pack_directory_to_dlc_file(directory_path, meta, "pretrain_tmp.sf_dlc")
        print("packed")
        system("explorer .")

    def __init__(self) -> None:
        gr.Markdown("## 打包为预训练 DLC")
        pack_to_pretrain_dlc_input_path = gr.Textbox(label="输入目录路径")
        pack_to_pretrain_dlc_model_name = gr.Textbox(label="模型名称")
        pack_to_pretrain_dlc_btn = gr.Button("打包为预训练 DLC")

        pack_to_pretrain_dlc_btn.click(
            self.pack_pretrain_dlc,
            inputs=[
                pack_to_pretrain_dlc_input_path,
                pack_to_pretrain_dlc_model_name,
            ],
        )
