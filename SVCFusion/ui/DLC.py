import gradio as gr

from SVCFusion.i18n import I


class DLC:
    def __init__(self):
        gr.Markdown()

        self.upload_file = gr.File(
            label=I.DLC.dlc_install_label,
            type="filepath",
        )
        self.install_btn = gr.Button(I.DLC.dlc_install_btn_value)
