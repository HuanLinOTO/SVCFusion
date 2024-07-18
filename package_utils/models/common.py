import os
from package_utils.i18n import I
from package_utils.uvr import getVocalAndInstrument
import gradio as gr

common_infer_form = {
    "audio": {
        "type": "audio",
        "label": I.common_infer.audio_label,
    },
    "use_vocal_remove": {
        "type": "checkbox",
        "info": I.common_infer.use_vocal_remove_info,
        "default": False,
        "label": I.common_infer.use_vocal_remove_label,
        "individual": True,
    },
    "use_remove_harmony": {
        "type": "checkbox",
        "info": I.common_infer.use_harmony_remove_info,
        "default": False,
        "label": I.common_infer.use_harmony_remove_label,
        "individual": True,
    },
    "f0": {
        "type": "dropdown",
        "info": I.common_infer.f0_info,
        "choices": [
            "parselmouth",
            "dio",
            "harvest",
            "crepe",
            "rmvpe",
            "fcpe",
        ],
        "default": "fcpe",
        "label": I.common_infer.f0_label,
    },
    "keychange": {
        "type": "slider",
        "max": 20,
        "min": -20,
        "default": 0,
        "step": 1,
        "info": I.common_infer.keychange_info,
        "label": I.common_infer.keychange_label,
    },
    "threshold": {
        "type": "slider",
        "max": 0,
        "min": -100,
        "default": -60,
        "step": 1,
        "label": I.common_infer.threshold_label,
        "info": I.common_infer.threshold_info,
    },
}

diff_based_infer_form = {
    "method": {
        "type": "dropdown",
        "info": I.diff_based_infer.method_info,
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": I.diff_based_infer.method_label,
    },
    "infer_step": {
        "type": "slider",
        "max": 100,
        "min": 1,
        "default": 50,
        "step": 1,
        "label": I.diff_based_infer.infer_step_label,
        "info": I.diff_based_infer.infer_step_info,
    },
    "t_start": {
        "type": "slider",
        "max": 1,
        "min": 0.1,
        "default": 0.7,
        "step": 0.1,
        "label": I.diff_based_infer.t_start_label,
        "info": I.diff_based_infer.t_start_info,
    },
}

common_preprocess_form = {
    "encoder": {
        "type": "dropdown",
        "info": I.common_preprocess.encoder_info,
        "choices": [
            # "hubertsoft",
            # "hubertbase",
            # "hubertbase768",
            # "contentvec",
            # "contentvec768",
            "contentvec768l12",
            # "cnhubertsoftfish",
        ],
        "default": "contentvec768l12",
        "label": I.common_preprocess.encoder_label,
    },
    "f0": {
        "type": "dropdown",
        "info": I.common_preprocess.f0_info,
        "choices": [
            "parselmouth",
            "dio",
            "harvest",
            "crepe",
            "rmvpe",
            "fcpe",
        ],
        "default": "fcpe",
        "label": I.common_preprocess.f0_label,
    },
    "device": {
        "type": "device_chooser",
    },
    # "force_cut": {
    #     "type": "checkbox",
    #     "info": "是否使用强制切片 (看不懂默认就对了)",
    #     "default": True,
    #     "label": "强制切片",
    # },
}

diff_based_preprocess_form = {
    "method": {
        "type": "dropdown",
        "info": I.diff_based_preprocess.method_info,
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": I.diff_based_preprocess.method_label,
    }
}


def infer_fn_proxy(fn):
    def infer_fn(params, progress):
        if params["use_vocal_remove"]:
            vocal, inst = getVocalAndInstrument(params["audio"], progress=progress)
            params["audio"] = vocal
            if params["use_remove_harmony"]:
                pass

        res = fn(params, progress=progress)

        return (
            gr.update(value=res),
            gr.update(
                visible=params["use_vocal_remove"],
                value=inst if params["use_vocal_remove"] else "tmp.wav",
            ),
        )

    return infer_fn


def train_fn_proxy(fn):
    def train_fn(params, progress):
        if os.path.exists("exp/workdir/stop.txt"):
            # del it
            os.remove("exp/workdir/stop.txt")
        res = fn(params, progress=progress)
        return gr.update(value=res)

    return train_fn
