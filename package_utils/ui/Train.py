import os
import random

import shutil
import time
import gradio as gr

from package_utils.const_vars import FOUZU, WORK_DIR_PATH
from package_utils.i18n import I
from package_utils.model_utils import detect_current_model_by_dataset
from package_utils.models.inited import (
    train_form,
    model_name_list,
    train_models_dict,
)
from package_utils.ui.Form import Form


class Train:
    def on_change_train_model(self):
        model_name = model_name_list[detect_current_model_by_dataset()]
        return gr.update(
            choices=train_models_dict[model_name],
            value=train_models_dict[model_name][0],
        )

    def archieve(self):
        gr.Info(I.train.archieving_tip)
        # 将 {WORKDIR} 移动到 archieves/yyyy-mm-dd-HH-MM-SS
        dst = f"archieve/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        shutil.move(WORK_DIR_PATH, dst)
        os.mkdir(WORK_DIR_PATH)
        if os.path.exists(f"{dst}/config.yaml"):
            shutil.copy(f"{dst}/config.yaml", f"{WORK_DIR_PATH}/config.yaml")
        if os.path.exists(f"{dst}/config.json"):
            shutil.copy(f"{dst}/config.json", f"{WORK_DIR_PATH}/config.json")
        # 用explorer打开归档目录
        os.system(f"explorer {os.path.abspath(dst)}")
        gr.Info(I.train.archieved_tip)

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
                    value=self.on_change_train_model,
                    interactive=True,
                )
                Form(
                    triger_comp=sub_model_type_dropdown,
                    models=train_form,
                )

                with gr.Row():
                    archieve_btn = gr.Button(
                        I.train.archieve_btn_value,
                        variant="stop",
                    )
                    stop_btn = gr.Button(
                        I.train.stop_btn_value,
                        variant="stop",
                    )

            train_model_type.change(
                self.on_change_train_model,
                outputs=[sub_model_type_dropdown],
            )

            archieve_btn.click(
                self.archieve,
            )
            stop_btn.click(self.stop)
