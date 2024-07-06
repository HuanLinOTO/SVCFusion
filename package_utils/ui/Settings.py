from re import M
from package_utils.config import JSONReader, applyChanges
from package_utils.i18n import I
from package_utils.config import system_config
from package_utils.locale import text_to_locale
from package_utils.ui.Form import Form

import gradio as gr

from package_utils.ui.FormTypes import FormDict


class Settings:
    def get_save_config_fn(self, prefix):
        def fn(config, progress=None):
            tmp = {
                prefix: config,
            }
            applyChanges(
                "configs/svcfusion.json",
                tmp,
                no_skip=True,
            )
            gr.Info(I.settings.saved_tip)

        return fn

    def __init__(self):
        ddsp6_pretrain_models = [
            I.settings.ddsp6.default_pretrained_model,
        ]

        self.form: FormDict = {
            I.settings.pkg_settings_label: {
                "form": {
                    "lang": {
                        "label": I.settings.pkg.lang_label,
                        "type": "dropdown",
                        "info": I.settings.pkg.lang_info,
                        "choices": text_to_locale.keys(),
                        "default": lambda: system_config.pkg.lang,
                    },
                },
                "callback": self.get_save_config_fn("pkg"),
            },
            I.settings.sovits_settings_label: {
                "form": {
                    "resolve_port_clash": {
                        "type": "checkbox",
                        "label": I.settings.sovits.resolve_port_clash_label,
                        "info": I.settings.sovits.resolve_port_clash_label,
                        "default": lambda: system_config.sovits.resolve_port_clash,
                    }
                },
                "callback": self.get_save_config_fn("sovits"),
            },
            I.settings.ddsp6_settings_label: {
                "form": {
                    "pretrained_model_preference": {
                        "type": "dropdown",
                        "label": I.settings.ddsp6.pretrained_model_preference_dropdown_label,
                        "choices": ddsp6_pretrain_models,
                        "value_type": "index",
                        "default": lambda: ddsp6_pretrain_models[
                            system_config.ddsp6.pretrained_model_preference
                        ],
                    },
                },
                "callback": self.get_save_config_fn("ddsp6"),
            },
        }

        self.triger = gr.Dropdown(
            label=I.settings.page,
            choices=[
                I.settings.pkg_settings_label,
                I.settings.sovits_settings_label,
                I.settings.ddsp6_settings_label,
            ],
            value=I.settings.pkg_settings_label,
            interactive=True,
        )

        Form(
            triger_comp=self.triger,
            models=self.form,
        )
