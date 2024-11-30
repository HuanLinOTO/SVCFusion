import os
from shutil import rmtree

from SVCFusion.config import JSONReader, YAMLReader
from SVCFusion.model_utils import detect_current_model_by_path
from .exec import executable
from loguru import logger
import yaml

from SVCFusion.const_vars import SUPPORT_MODEL_TYPE, TYPE_INDEX_TO_CONFIG_NAME
from SVCFusion.file import make_dirs
from SVCFusion.exec import exec
import gradio as gr
from ddspsvc.draw import main as draw_main


class DrawArgs:
    val = "data/val/audio"
    sample_rate = 1
    train = "data/train/audio"
    extensions = ["wav", "flac"]


class PreprocessArgs:
    config = "configs/ddsp_reflow.yaml"
    device = "cuda"


def resample(src, dst):
    assert (
        exec(
            f"{executable} fap/__main__.py resample {src} {dst} --mono",
        )
        == 0
    ), "重采样失败，请截图日志反馈，日志在上面 不在这里！！"


def to_wav(src, dst):
    assert (
        exec(
            f"{executable} fap/__main__.py to-wav {src} {dst}",
        )
        == 0
    ), "转 WAV 失败，请截图日志反馈，日志在上面 不在这里！！"


def slice_audio(src, dst, max_duration):
    assert (
        exec(
            f"{executable} fap/__main__.py slice-audio-v2 {src} {dst} --max-duration {max_duration} --flat-layout --merge-short --clean"
        )
        == 0
    ), "切割音频失败，请截图日志反馈，日志在上面 不在这里！！"


def auto_normalize_dataset(
    output_dir: str, rename_by_index: bool, progress: gr.Progress
):
    make_dirs(output_dir, True)
    resample(
        "dataset_raw/",
        "tmp/resampled/",
    )
    slice_audio(
        "tmp/resampled/",
        output_dir,
        max_duration=15.0,
    )
    rmtree("tmp/resampled")
    if rename_by_index:
        for i, spk in enumerate(
            [i for i in os.listdir(output_dir) if i != ".ipynb_checkpoints"]
        ):
            os.rename(f"{output_dir}/{spk}", f"{output_dir}/{i + 1}")


def check_spks():
    spks = []
    for f in os.listdir("dataset_raw"):
        if os.path.isdir(os.path.join("dataset_raw", f)):
            spks.append(f)
    return spks


def get_spk_from_dir(search_path):
    model_type_index = detect_current_model_by_path(search_path)
    if model_type_index == 2:
        with JSONReader(f"{search_path}/config.json") as config:
            return list(config["spk"].keys())
    elif model_type_index in [0, 1, 3]:
        with YAMLReader(f"{search_path}/config.yaml") as config:
            return config["spks"]


def auto_preprocess(
    f0="fcpe",
    encoder="euler",
    device="cuda",
    use_slice_audio=True,
    max_duration=15,
    model_type_index=0,
):
    make_dirs("tmp/resampled", True)

    config_name = TYPE_INDEX_TO_CONFIG_NAME[model_type_index]

    # 复制 config/ddsp_reflow.yaml.template -> ddsp_reflow.yaml
    # shutil.copy("configs/ddsp_reflow.yaml.template", "configs/ddsp_reflow.yaml")

    spks = []
    for f in os.listdir("dataset_raw"):
        if os.path.isdir(os.path.join("dataset_raw", f)):
            spks.append(f)
        # 扫描角色目录，如果发现 .WAV 文件 改成 .wav
        for root, dirs, files in os.walk(f"dataset_raw/{f}"):
            for file in files:
                if file.endswith(".WAV"):
                    logger.info(f"Renamed {file} to {file.replace('.WAV', '.wav')}")
                    os.rename(
                        os.path.join(root, file),
                        os.path.join(root, file.replace(".WAV", ".wav")),
                    )

    # 修改 config[data][f0_extractor] 为 f0
    with open("configs/" + config_name, "r") as f:
        config = yaml.safe_load(f)
        config["data"]["f0_extractor"] = f0
        config["data"]["encoder"] = encoder
        config["model"]["n_spk"] = len(spks)
        config["spks"] = spks

    with open("configs/" + config_name, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open("configs/ddsp_reflow.yaml", "r") as f:
        config = yaml.safe_load(f)
        config["spks"] = spks

    with open("configs/ddsp_reflow.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    make_dirs("data/train/", True)
    make_dirs("data/train/audio", True)
    make_dirs("data/val/", True)
    make_dirs("data/val/audio", True)

    logger.info("Resample started")
    resample(
        "dataset_raw/", "tmp/resampled/" if use_slice_audio else "data/train/audio/"
    )
    logger.info("Resample finished")

    if use_slice_audio:
        slice_audio("tmp/resampled/", "data/train/audio/", max_duration)

    # 将 data/train/audio/ 下面的文件夹按照出现顺序命名
    for i, spk in enumerate(spks):
        os.rename(f"data/train/audio/{spk}", f"data/train/audio/{i + 1}")

    logger.info("Drawing datasets")
    draw_main(DrawArgs())
    logger.info("Draw finished")

    logger.info("Preprocess started")

    type_index = model_type_index

    if type_index == 0:
        exec(f"{executable} -m ddspsvc.preprocess -c configs/{config_name} -d {device}")
    elif type_index == 1:
        exec(
            f"{executable} -m ReFlowVaeSVC.preprocess -c configs/{config_name} -d {device}"
        )
    elif type_index == 2:
        exec(
            f"{executable} -m SoVITS.preprocess_new -c configs/sovits.json -d {device}"
        )
    elif type_index == 3:
        exec(
            f"{executable} -m SoVITS.preprocess_new -c configs/sovits_diff.yaml -d {device} --use_diff"
        )
    logger.info("Preprocess finished")

    # 写入 data/model_type
    with open("data/model_type", "w") as f:
        f.write(SUPPORT_MODEL_TYPE[type_index])
