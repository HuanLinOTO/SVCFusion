from datetime import datetime
import os
import shutil
import gradio as gr

from package_utils.config import JSONReader, YAMLReader
from package_utils.const_vars import (
    WORK_DIR_PATH,
)
from package_utils.file import make_dirs
from package_utils.exec import exec
from package_utils.i18n import I


def reload_models(search_dir):
    global models, search_paths
    models = search_models(search_paths[search_dir])
    search_paths = [
        WORK_DIR_PATH,
        *[
            "archieve/" + p
            for p in os.listdir("archieve")
            if os.path.isdir(os.path.join("archieve", p))
        ],
        *[
            "models/" + p
            for p in os.listdir("models")
            if os.path.isdir(os.path.join("models", p))
        ],
    ]
    return (
        gr.update(choices=models, value=models[-1]),
        gr.update(
            choices=[
                "工作目录",
                *[
                    p.replace("models/", "models 文件夹 - ").replace(
                        "archieve/", "已归档训练 - "
                    )
                    for p in search_paths
                    if not p.startswith("exp")
                ],
            ],
        ),
    )


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


def archieve():
    make_dirs("archieve/", False)
    path = f"./archieve/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    # make_dirs(path, False)
    shutil.move(WORK_DIR_PATH, path)
    make_dirs(WORK_DIR_PATH, False)
    if os.name == "nt":
        exec("explorer " + path.replace("/", "\\"))


def load_pretrained(model_name, extra):
    pretrained_path = os.path.join("pretrained", model_name, extra)
    if model_name != "sovits_diff":
        dst = WORK_DIR_PATH
    else:
        dst = os.path.join(WORK_DIR_PATH, "diffusion")
    for file in os.listdir(pretrained_path):
        if "config.yaml" in file:
            continue
        shutil.copy(os.path.join(pretrained_path, file), dst)

    # 如果pretrain目录下面有 config.yaml，读取并返回
    if os.path.exists(os.path.join(pretrained_path, "config.yaml")):
        with YAMLReader(os.path.join(pretrained_path, "config.yaml")) as config:
            return config


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
