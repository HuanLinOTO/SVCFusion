from datetime import datetime
import os
import shutil
import gradio as gr
from networkx import planted_partition_graph

from SVCFusion.config import JSONReader, YAMLReader
from SVCFusion.const_vars import (
    WORK_DIR_PATH,
)
from SVCFusion.file import make_dirs
from SVCFusion.exec import exec
from SVCFusion.i18n import I


def search_models(search_dir) -> list:
    models = []
    # for root, dirs, files in os.walk(search_dir):
    #     for file in files:
    #         if file.endswith(".pt"):
    #             models.append(file)
    for file in os.listdir(search_dir):
        if (
            file.endswith(".pt")
            and os.path.isfile(os.path.join(search_dir, file))
            and file != "model_0.pt"
        ):
            models.append(file)
    if len(models) == 0:
        models = ["无模型"]
    return models


def archive():
    make_dirs("archive/", False)
    path = f"./archive/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    # make_dirs(path, False)
    shutil.move(WORK_DIR_PATH, path)
    make_dirs(WORK_DIR_PATH, False)
    if os.name == "nt":
        exec("explorer " + path.replace("/", "\\"))


def get_pretrain_models(model_name):
    pretrain_models = []
    path_pretrain_store = os.path.join("pretrained", model_name)
    if not os.path.exists(path_pretrain_store):
        return pretrain_models

    for model in os.listdir(path_pretrain_store):
        path_meta = os.path.join(path_pretrain_store, model, "meta.yaml")
        if not os.path.exists(path_meta):
            continue
        with YAMLReader(path_meta) as meta:
            pretrain_models.append(meta)
        meta["_path"] = os.path.join(path_pretrain_store, model)
    return pretrain_models


def get_pretrain_models_meta(path):
    path_meta = os.path.join(path, "meta.yaml")
    if not os.path.exists(path_meta):
        raise FileNotFoundError(f"File not found: {path_meta}")
    with YAMLReader(path_meta) as meta:
        return meta


def get_pretrain_models_form_item(model_name):
    def update():
        models = get_pretrain_models(model_name)
        choices = []

        for model in models:
            choices.append(
                (
                    model.get("title"),
                    model.get("_path"),
                )
            )

        if len(choices) == 0:
            choices = [
                (I.train.pretrain_model_not_found_tip, ""),
            ]
        return gr.update(
            choices=choices,
            value=choices[0][1],
        )

    def update_tip(path):
        meta = get_pretrain_models_meta(path)
        infos = []

        if meta.get("official", False) is True:
            infos.append("<b>" + I.train.official_pretrain_model + "</b>")

        if meta.get("desc"):
            infos.append(meta["desc"])
        if meta.get("vec"):
            infos.append(I.train.pretrain_model_vec + ": " + meta["vec"])
        if meta.get("vocoder"):
            infos.append(I.train.pretrain_model_vocoder + ": " + meta["vocoder"])
        if meta.get("size"):
            infos.append(I.train.pretrain_model_size + ": " + meta["size"])
        if meta.get("attn"):
            infos.append(
                I.train.pretrain_model_attn + ": " + I.form.dorpdown_liked_checkbox_yes
                if meta["attn"]
                else I.form.dorpdown_liked_checkbox_no
            )
        return f'<div style="background: var(--block-background-fill); padding: 8px;">{"<br/>".join(infos)}</div>'

    return {
        "#pretrain": {
            "type": "dropdown",
            "default": update,
            "choices": [],
            "info": I.train.choose_pretrain_model_info,
            "label": I.train.choose_pretrain_model_label,
            "individual": True,
            "addition_tip_when_update": update_tip,
        }
    }


def load_pretrained(
    model_name, requirements: dict, path: str = None, scan_only: bool = False
) -> tuple[dict, bool]:
    """
    Load a pretrained model based on the given requirements.
    Args:
        model_name (str): The name of the model to load.
        requirements (dict): A dictionary of requirements that the pretrained model must meet.
        path (str, optional): The path to a specific pretrained model. Defaults to None.
        scan_only (bool, optional): If True, only scan for the model without copying files. Defaults to False.
    Returns:
        tuple[dict, bool]: A tuple where the first element is the configuration dictionary of the pretrained model,
                           and the second element is a boolean indicating whether the model was successfully loaded.
    """

    path_pretrain_store = os.path.join("pretrained", model_name)
    if not os.path.exists(path_pretrain_store):
        return {}, False

    finded_pretrained_path = path

    if not finded_pretrained_path:
        for model in os.listdir(path_pretrain_store):
            # 读取每个模型文件夹下面的 meta.yaml，找到符合要求的模型
            pretrained_path = os.path.join(path_pretrain_store, model)
            path_meta = os.path.join(pretrained_path, "meta.yaml")
            if not os.path.exists(path_meta):
                continue
            with YAMLReader(path_meta) as meta:
                if all(meta.get(key) == value for key, value in requirements.items()):
                    print(f"Pretrained model found: {model}")

                    finded_pretrained_path = pretrained_path

    if finded_pretrained_path:
        if model_name != "sovits_diff":
            dst = WORK_DIR_PATH
        else:
            dst = os.path.join(WORK_DIR_PATH, "diffusion")
        for file in os.listdir(finded_pretrained_path):
            if "config.yaml" in file:
                continue
            if scan_only:
                continue
            shutil.copy(os.path.join(finded_pretrained_path, file), dst)

        # 如果pretrain目录下面有 config.yaml，读取并返回
        if os.path.exists(os.path.join(finded_pretrained_path, "config.yaml")):
            with YAMLReader(
                os.path.join(finded_pretrained_path, "config.yaml")
            ) as config:
                return config, True
        return {}, True
    return {}, False


def tensorboard():
    # cmd = ".conda\\Scripts\\tensorboard --logdir=exp/"
    # subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", cmd])
    from tensorboard import program

    log_dir = "./exp"

    # 启动TensorBoard服务器
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir])
    url = tb.launch()
    return url


def detect_current_model_by_path(model_path, alert=False):
    # 读取 model_path/config.yaml 中的 model_type
    is_unknown = False
    if os.path.exists(model_path + "/config.json"):
        with JSONReader(model_path + "/config.json") as config:
            if config.get("model_type_index") is None:
                is_unknown = True
                model_type = -1
            else:
                is_unknown = False
                model_type = config["model_type_index"]
    elif os.path.exists(model_path + "/config.yaml"):
        # DDSP / ReflowVAE
        with YAMLReader(model_path + "/config.yaml") as config:
            if config.get("model_type_index") is None:
                is_unknown = True
                model_type = -1
            else:
                is_unknown = False
                model_type = config["model_type_index"]
    else:
        is_unknown = True
        model_type = -1

    if is_unknown and alert:
        gr.Info(I.unknown_model_type_tip)
    return model_type


def detect_current_model_by_dataset():
    # 读取 data/model_type 并返回内容
    try:
        with open("data/model_type", "r") as f:
            model_type = f.read()
        return int(model_type)
    except Exception as _e:
        # 写入 data/model_type "ddsp6"
        with open("data/model_type", "w") as f:
            f.write("0")
        return 0
