from package_utils.config import JSONReader, applyChanges
from package_utils.i18n import I
from package_utils.ui.Form import Form

import gradio as gr


class Settings:
    def get_config(*args):
        with JSONReader("configs/svcfusion.json") as config:
            return config

    def save_config(self, config, progress=None):
        # with JSONReader("configs/svcfusion.json") as c:
        #     c.update(config)
        applyChanges("configs/svcfusion.json", config, no_skip=True)
        gr.Info("已保存")

    def __init__(self):
        self.form = {
            I.settings.pkg_settings_label: {
                "form": {
                    "lang": {
                        "label": I.settings.lang_label,
                        "type": "dropdown",
                        "info": I.settings.lang_info,
                        "choices": ["简体中文", "English"],
                        "default": lambda: self.get_config()["lang"],
                    },
                },
                "callback": self.save_config,
            }
        }

        self.triger = gr.Dropdown(
            label="页面",
            choices=[
                I.settings.pkg_settings_label,
            ],
            value=I.settings.pkg_settings_label,
            interactive=True,
        )

        Form(
            triger_comp=self.triger,
            models=self.form,
        )
