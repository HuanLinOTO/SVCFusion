import json
import os
import shutil
import gradio as gr

from SVCFusion.config import JSONReader
from SVCFusion.i18n import I


class Compatible:
    def on_upload_sovits_model(
        self,
        model_name,
        sovits_main_model,
        sovits_diff_model,
        sovits_cluster_model,
        sovits_main_model_config,
        sovits_diff_model_config,
    ):
        model_dir = f"models/{model_name}"

        # 如果存在 model_dir 则提示已存在
        if os.path.exists(model_dir):
            gr.Info(I.compatible_models.model_exists)
            return

        if not all(
            [
                model_name,
                sovits_main_model,
                sovits_main_model_config,
            ]
        ):
            gr.Info(I.compatible_models.upload_error)
            return
        if (sovits_diff_model_config and not sovits_diff_model) or (
            sovits_diff_model and not sovits_diff_model_config
        ):
            gr.Info(I.compatible_models.upload_error)
            return

        os.makedirs(model_dir, exist_ok=True)
        for f, name in [
            (sovits_main_model, "model.pth"),
            (sovits_diff_model, "model.pt"),
            (sovits_main_model_config, "config.json"),
            (sovits_diff_model_config, "config.yaml"),
        ]:
            if f:
                shutil.copy(f, f"{model_dir}/{name}")

        if sovits_cluster_model:
            if sovits_cluster_model.endswith(".pt"):
                shutil.copy(sovits_cluster_model, f"{model_dir}/kmeans_10000.pt")
            else:
                shutil.copy(sovits_cluster_model, f"{model_dir}/feature_and_index.pkl")

        # 打开 main config 往里面加一条 model_type_index = 2
        with JSONReader(f"{model_dir}/config.json") as config:
            config["model_type_index"] = 2
        with open(f"{model_dir}/config.json", "w") as f:
            f.write(json.dumps(config, indent=4, ensure_ascii=False))

        gr.Info(I.compatible_models.upload_success)

    def __init__(self) -> None:
        with gr.Tabs():
            model_name = gr.Textbox(
                label=I.compatible_models.model_name_label,
            )

            with gr.TabItem(I.compatible_models.compatible_sovits):
                sovits_main_model = gr.File(
                    label=I.compatible_models.sovits_main_model_label
                )

                sovits_diff_model = gr.File(
                    label=I.compatible_models.sovits_diff_model_label
                )

                sovits_cluster_model = gr.File(
                    label=I.compatible_models.sovits_cluster_model_label
                )

                sovits_main_model_config = gr.File(
                    label=I.compatible_models.sovits_main_model_config_label
                )

                sovits_diff_model_config = gr.File(
                    label=I.compatible_models.sovits_diff_model_config_label
                )

                sovits_submit_btn = gr.Button(I.form.submit_btn_value)

                sovits_submit_btn.click(
                    self.on_upload_sovits_model,
                    inputs=[
                        model_name,
                        sovits_main_model,
                        sovits_diff_model,
                        sovits_cluster_model,
                        sovits_main_model_config,
                        sovits_diff_model_config,
                    ],
                )
