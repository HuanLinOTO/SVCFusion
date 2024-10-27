import os
import random

import shutil
import time
import gradio as gr

from fap.utils.file import make_dirs
from SVCFusion import model_utils
from SVCFusion.const_vars import FOUZU, WORK_DIR_PATH
from SVCFusion.i18n import I
from SVCFusion.model_utils import detect_current_model_by_dataset
from SVCFusion.models.inited import (
    train_form,
    model_name_list,
    train_models_dict,
)
from SVCFusion.ui.Form import Form
import webbrowser


class Train:
    def on_change_train_model(self, sub_model_type=None):
        model_name = model_name_list[detect_current_model_by_dataset()]

        return (
            gr.update(
                choices=train_models_dict[model_name],
                value=train_models_dict[model_name][0],
            ),
            self.update_pretrain_model_chooser(sub_model_type)
            if sub_model_type
            else gr.update(),
        )

    def get_sub_model_type_dropdown_value(self):
        return self.on_change_train_model()[0]

    def update_pretrain_model_chooser(self, sub_model_type):
        pretrained_models = model_utils.get_pretrain_models(sub_model_type)

        pretrained_models_choices = []
        for model in pretrained_models:
            pretrained_models_choices.append(
                (
                    model["name"],
                    model["_path"],
                )
            )
        return (
            gr.update(
                choices=pretrained_models_choices,
                value=pretrained_models[0] if len(pretrained_models) > 0 else None,
            ),
        )

    tb_url = ""

    def launch_tb(self):
        gr.Info(I.train.launching_tb_tip)
        if not self.tb_url:
            self.tb_url = model_utils.tensorboard()

        webbrowser.open(self.tb_url)
        gr.Info(I.train.launched_tb_tip.replace("{1}", self.tb_url))

    def archive(self):
        gr.Info(I.train.archieving_tip)
        # 将 {WORKDIR} 移动到 archives/yyyy-mm-dd-HH-MM-SS
        dst = f"archive/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        shutil.move(WORK_DIR_PATH, dst)
        os.mkdir(WORK_DIR_PATH)
        if os.path.exists(f"{dst}/config.yaml"):
            shutil.copy(f"{dst}/config.yaml", f"{WORK_DIR_PATH}/config.yaml")
        if os.path.exists(f"{dst}/config.json"):
            shutil.copy(f"{dst}/config.json", f"{WORK_DIR_PATH}/config.json")
        if os.path.exists(f"{dst}/diffusion/config.yaml"):
            make_dirs(f"{WORK_DIR_PATH}/diffusion")
            shutil.copy(
                f"{dst}/diffusion/config.yaml",
                f"{WORK_DIR_PATH}/diffusion/config.yaml",
            )
        # 用explorer打开归档目录
        os.system(f"explorer {os.path.abspath(dst)}")
        gr.Info(I.train.archived_tip)

    def stop(self):
        # 往 workdir 写入 stop.txt
        with open(f"{WORK_DIR_PATH}/stop.txt", "w") as f:
            f.write("stop")
        gr.Info(I.train.stopped_tip)

    def __init__(self) -> None:
        with gr.Row():
            with gr.Column(scale=1):
                train_model_type = gr.Textbox(
                    label=I.train.current_train_model_label,
                    value=lambda: model_name_list[detect_current_model_by_dataset()],
                    every=1,
                )

                fouzu = gr.Markdown(value=FOUZU)
                gd_plus_1 = gr.Button(
                    I.train.gd_plus_1,
                )

                def gd_plus_1_fn():
                    gr.Info(I.train.gd_plus_1_tip)
                    for i in range(3):
                        yield "".join(
                            [
                                "*"
                                if c == " " and random.choice([True, False * 20])
                                else c
                                for c in FOUZU
                            ]
                        )
                        time.sleep(1)
                    yield FOUZU

                gd_plus_1.click(gd_plus_1_fn, outputs=[fouzu])
            with gr.Column(scale=4):
                sub_model_type_dropdown = gr.Dropdown(
                    label=I.train.choose_sub_model_label,
                    value=self.get_sub_model_type_dropdown_value,
                    interactive=True,
                )

                pretrain_model_dropdown = gr.Dropdown(
                    label=I.train.choose_pretrain_model_label,
                    interactive=True,
                )

                # debug: print
                pretrain_model_dropdown.change(
                    lambda x: print(x),
                    inputs=[pretrain_model_dropdown],
                )

                Form(
                    triger_comp=sub_model_type_dropdown,
                    models=train_form,
                    submit_btn_text=I.train.start_train_btn_value,
                )

                with gr.Row():
                    archive_btn = gr.Button(
                        I.train.archive_btn_value,
                        variant="stop",
                    )
                    stop_btn = gr.Button(
                        I.train.stop_btn_value,
                        variant="stop",
                    )
                with gr.Row():
                    tensorboard_btn = gr.Button(
                        I.train.tensorboard_btn,
                        variant="primary",
                    )

            train_model_type.change(
                self.on_change_train_model,
                inputs=[
                    sub_model_type_dropdown,
                ],
                outputs=[
                    sub_model_type_dropdown,
                    pretrain_model_dropdown,
                ],
            )

            sub_model_type_dropdown.change(
                self.update_pretrain_model_chooser,
                inputs=[
                    sub_model_type_dropdown,
                ],
                outputs=[
                    pretrain_model_dropdown,
                ],
            )

            archive_btn.click(
                self.archive,
            )
            stop_btn.click(self.stop)

            tensorboard_btn.click(self.launch_tb)
