import gradio as gr

from SVCFusion.dlc import install_dlc
from SVCFusion.i18n import I
from loguru import logger

support_ext = [".sf_dlc"]


class DLC:
    def install(self, upload_file: str):
        gr.Info(I.DLC.dlc_installing_tip)

        if not upload_file:
            raise gr.Error(I.DLC.dlc_install_empty)
        if not upload_file.endswith(tuple(support_ext)):
            raise gr.Error(I.DLC.dlc_install_ext_error)

        logger.info(f"Install DLC {upload_file}")
        success = install_dlc(upload_file)
        logger.info(f"Install {upload_file} {'success' if success else 'fail'}")

        if success:
            gr.Info(I.DLC.dlc_install_success)
        else:
            raise gr.Error(I.DLC.dlc_install_fail)

    def __init__(self):
        gr.Markdown()

        self.upload_file = gr.File(
            label=I.DLC.dlc_install_label,
            type="filepath",
        )
        self.install_btn = gr.Button(I.DLC.dlc_install_btn_value)

        self.install_btn.click(
            self.install,
            inputs=[self.upload_file],
        )
