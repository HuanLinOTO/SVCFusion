import gc
import hashlib
import json
import os
import shutil
from traceback import print_exception

import torch
import torchaudio
from SVCFusion.const_vars import EMPTY_WAV_PATH
from SVCFusion.i18n import I
from SVCFusion.uvr import getVocalAndInstrument
import gradio as gr

common_infer_form = {
    "audio": {
        "type": "audio",
        "label": I.common_infer.audio_label,
        "individual": True,
    },
    "audio_batch": {
        "type": "file",
        "label": I.common_infer.audio_label,
        "visible": False,
        "individual": True,
    },
    "use_batch": {
        "type": "show_switch",
        "label": I.common_infer.use_batch_label,
        "default": False,
        "individual": True,
        "default_show": ["audio"],
        "other_show": ["audio_batch"],
    },
    "use_vocal_separation": {
        "type": "checkbox",
        "info": I.common_infer.use_vocal_separation_info,
        "default": False,
        "label": I.common_infer.use_vocal_separation_label,
        "individual": True,
    },
    "use_de_reverb": {
        "type": "checkbox",
        "info": I.common_infer.use_de_reverb_info,
        "default": False,
        "label": I.common_infer.use_de_reverb_label,
        "individual": True,
    },
    "use_harmonic_remove": {
        "type": "checkbox",
        "info": I.common_infer.use_harmonic_remove_info,
        "default": False,
        "label": I.common_infer.use_harmonic_remove_label,
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

ddsp_based_infer_form = {
    "method": {
        "type": "dropdown",
        "info": I.ddsp_based_infer.method_info,
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": I.ddsp_based_infer.method_label,
    },
    "infer_step": {
        "type": "slider",
        "max": 100,
        "min": 1,
        "default": 50,
        "step": 1,
        "label": I.ddsp_based_infer.infer_step_label,
        "info": I.ddsp_based_infer.infer_step_info,
    },
    "t_start": {
        "type": "slider",
        "max": 1,
        "min": 0.1,
        "default": 0.7,
        "step": 0.1,
        "label": I.ddsp_based_infer.t_start_label,
        "info": I.ddsp_based_infer.t_start_info,
    },
    "num_formant_shift_key": {
        "type": "slider",
        "max": 10,
        "min": -10,
        "default": 0,
        "step": 0.1,
        "label": I.ddsp_based_infer.num_formant_shift_key_label,
        "info": I.ddsp_based_infer.num_formant_shift_key_info,
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
}

ddsp_based_preprocess_form = {
    "method": {
        "type": "dropdown",
        "info": I.ddsp_based_preprocess.method_info,
        "choices": ["euler", "rk4"],
        "default": "euler",
        "label": I.ddsp_based_preprocess.method_label,
    },
}


def infer_fn_proxy(fn):
    def infer_fn(params, progress):
        if not params["use_batch"]:
            params["audio"] = [params["audio"]]
        else:
            params["audio"] = params["audio_batch"]
        result = []
        inst_list = []
        for audio in params["audio"]:
            processed_vocal = False
            processed_inst = False

            try:
                wf, sr = torchaudio.load(audio)
                # 重采样到 44100,单声道
                if wf.size(0) > 1:
                    wf = wf.mean(0, keepdim=True)
                if sr != 44100:
                    wf = torchaudio.transforms.Resample(sr, 44100)(wf)
                torchaudio.save(audio, wf, 44100)

                if (
                    params["use_vocal_separation"]
                    or params["use_de_reverb"]
                    or params["use_harmonic_remove"]
                ):
                    vocal, inst = getVocalAndInstrument(
                        audio,
                        use_vocal_fetch=params["use_vocal_separation"],
                        use_de_reverb=params["use_de_reverb"],
                        use_harmonic_remove=params["use_harmonic_remove"],
                        progress=progress,
                    )
                    audio = vocal
                    inst_list.append(inst)
                    processed_inst = True

                new_params = {}
                new_params.update(params)
                new_params["audio"] = audio
                new_params["hash"] = hashlib.md5(
                    json.dumps(new_params).encode()
                ).hexdigest()

                res = fn(new_params, progress=progress)

                result.append(res)
                processed_vocal = True

            except Exception as e:
                gr.Info(
                    I.error_when_infer.replace("{1}", str(audio)).replace("{2}", str(e))
                )
                print_exception(e)
                if not processed_inst:
                    inst_list.append(EMPTY_WAV_PATH)
                if not processed_vocal:
                    result.append(EMPTY_WAV_PATH)

        moved_vocal = []
        moved_inst = []

        for index in range(len(params["audio"])):
            vocal = result[index]
            raw = params["audio"][index]
            filename = os.path.basename(raw)
            filename = filename[: filename.rfind(".")]
            # mixed_file = mixAudio(
            #     vocal=vocal,
            #     inst=inst,
            #     room_length=1,
            #     room_width=1,
            #     room_height=1,
            #     vocal_db=13,
            # )

            vocal_dst = f"tmp/total_opt/inst/{filename}.wav"
            shutil.copy(vocal, vocal_dst)
            moved_vocal.append(vocal_dst)

            if params["use_vocal_separation"]:
                inst = inst_list[index]
                inst_dst = f"tmp/total_opt/vocal/{filename}.wav"
                shutil.copy(inst, inst_dst)
                moved_inst.append(inst_dst)

            # wf, sr = torchaudio.load(mixed_file)

            # wf = (
            #     (wf + torch.randn(wf.size()) * random.uniform(0, 0.01))
            #     if random.random() > 0.5
            #     else wf
            # )
            # torchaudio.save(f"tmp/total_opt/{filename}.wav", wf, sr)

        gc.collect()
        torch.cuda.empty_cache()

        return (
            gr.update(
                value=EMPTY_WAV_PATH if params["use_batch"] else moved_vocal[0],
                visible=not params["use_batch"],
            ),
            gr.update(
                visible=params["use_vocal_separation"] and not params["use_batch"],
                value=(
                    EMPTY_WAV_PATH
                    if params["use_batch"] or not params["use_vocal_separation"]
                    else moved_inst[0]
                ),
            ),
            gr.update(
                value=(moved_vocal if params["use_batch"] else EMPTY_WAV_PATH),
                visible=params["use_batch"],
            ),
            gr.update(
                value=(
                    moved_inst
                    if params["use_batch"] and params["use_vocal_separation"]
                    else EMPTY_WAV_PATH
                ),
                visible=params["use_vocal_separation"] and params["use_batch"],
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


def preprocess_fn_proxy(fn):
    def preprocess_fn(params, progress):
        # 删掉 exp/workdir/
        if os.path.exists("exp/workdir"):
            shutil.rmtree("exp/workdir")
        # 补回去一个文件夹
        os.makedirs("exp/workdir")

        res = fn(params, progress=progress)
        return gr.update(value=res)

    return preprocess_fn


__all__ = [
    "common_infer_form",
    "ddsp_based_infer_form",
    "common_preprocess_form",
    "ddsp_based_preprocess_form",
    "infer_fn_proxy",
    "train_fn_proxy",
    "preprocess_fn_proxy",
]
