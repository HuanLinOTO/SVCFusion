import gc
import hashlib
import os
from pathlib import Path
import traceback
import gradio as gr

from Music_Source_Separation_Training import inference as msst_inference

from SoVITS import logger
import torch
from SVCFusion.config import system_config
from SVCFusion.i18n import I
from vr import AudioPre, AudioPreDeEcho

os.environ["PATH"] += os.pathsep + os.getcwd()

device = "cuda"
is_half = True


def preprocess_path(path):
    return path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def uvr(
    model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, output
):
    try:
        inp_root = preprocess_path(inp_root)
        save_root_vocal = preprocess_path(save_root_vocal)
        save_root_ins = preprocess_path(save_root_ins)
        is_hp3 = "HP3" in model_name

        func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
        pre_fun = func(
            agg=int(agg),
            model_path=os.path.join("other_weights", model_name + ".pth"),
            device=device,
            is_half=is_half,
        )
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            print(inp_path)
            if os.path.isfile(inp_path) is False:
                print(f"File {inp_path} not found")
                continue
            # need_reformat = 1
            done = 0
            # try:
            #     info = ffmpeg.probe(inp_path, cmd="ffprobe")
            #     if (
            #         info["streams"][0]["channels"] == 2
            #         and info["streams"][0]["sample_rate"] == "44100"
            #     ):
            #         need_reformat = 0
            #         pre_fun._path_audio_(
            #             inp_path,
            #             save_root_ins,
            #             save_root_vocal,
            #             format0,
            #             is_hp3,
            #             output,
            #         )
            #         done = 1
            # except Exception:
            #     need_reformat = 1
            #     traceback.print_exc()
            # if need_reformat == 1:
            #     tmp_path = "%s/%s.reformatted.wav" % (
            #         os.path.join("tmp"),
            #         os.path.basename(inp_path),
            #     )
            #     os.system(
            #         'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
            #         % (inp_path, tmp_path)
            #     )
            #     inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path,
                        save_root_ins,
                        save_root_vocal,
                        format0,
                        is_hp3,
                        output,
                    )
            except Exception:
                traceback.print_exc()
    except Exception:
        print(traceback.format_exc())
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MSSTArgs:
    input_folder = None
    store_dir = None
    model_type = None

    def __init__(self, input_folder, store_dir, model_type) -> None:
        self.input_folder = input_folder
        self.store_dir = store_dir
        self.model_type = model_type


model_type_to_info = {
    "bs_roformer": {
        "config": "Music_Source_Separation_Training/configs/model_bs_roformer_ep_368_sdr_12.9628.yaml",
        "model": "other_weights/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "real_type": "bs_roformer",
    },
    "deverb_mel_band_roformer": {
        "config": r"Music_Source_Separation_Training/configs/deverb_mel_band_roformer.yaml",
        "model": "other_weights/deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt",
        "real_type": "mel_band_roformer",
    },
    "deverb_bs_roformer": {
        "config": r"Music_Source_Separation_Training/configs/deverb_bs_roformer_8_256dim_8depth.yaml",
        "model": "other_weights/deverb_bs_roformer_8_256dim_8depth.ckpt",
        "real_type": "bs_roformer",
    },
    "karaoke_mel_band_roformer": {
        "config": r"Music_Source_Separation_Training/configs/config_mel_band_roformer_karaoke.yaml",
        "model": "other_weights/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "real_type": "mel_band_roformer",
    },
    "kim_vocals_mel_band_roformer": {
        "config": r"Music_Source_Separation_Training/configs/kim_config_vocals_mel_band_roformer.yaml",
        "model": "other_weights/KimMelBandRoformer.ckpt",
        "real_type": "mel_band_roformer",
    },
}

job_to_model_type = {
    "vocal": "bs_roformer",
    "kim_vocal": "kim_vocals_mel_band_roformer",
    "deverb": "deverb_bs_roformer",
    "karaoke": "karaoke_mel_band_roformer",
}


def run_msst(
    inp_path,
    inp_hash,
    vocal_opt_path,
    inst_opt_path,
    progress=gr.Progress(track_tqdm=True),
    real_type="bs_roformer",
    model_type="bs_roformer",
    progress_desc: str = "",
    save_inst=True,
):
    vocal_path = f"./tmp/msst_opt/{inp_hash}_Vocals.wav"
    inst_path = f"./tmp/msst_opt/{inp_hash}_Instrument.wav"

    args = MSSTArgs(
        input_folder=inp_path,
        store_dir=f"./tmp/msst_opt/{inp_hash}",
        model_type="bs_roformer",
    )
    msst_model, bsroformer_config = msst_inference.get_model_from_config(
        real_type,
        model_type_to_info[model_type]["config"],
    )
    model_path = model_type_to_info[model_type]["model"]
    print("Start from checkpoint: {}".format(model_path))

    if torch.cuda.is_available():
        device = torch.device(system_config.infer.msst_device)
    else:
        logger.info(
            "CUDA is not avilable. Run inference on CPU. It will be very slow..."
        )
        device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=device)
    if args.model_type == "htdemucs":
        # Fix for htdemucs pround etrained models
        if "state" in state_dict:
            state_dict = state_dict["state"]
    msst_model.load_state_dict(state_dict)
    msst_model.to(device)

    msst_inference.run_folder(
        model=msst_model,
        args=args,
        vocal_opt_path=vocal_opt_path,
        inst_opt_path=inst_opt_path,
        config=bsroformer_config,
        device=device,
        progress=progress,
        save_inst=save_inst,
        progress_desc=progress_desc,
    )
    del msst_model
    torch.cuda.empty_cache()
    gc.collect()

    return vocal_path, inst_path


def getVocalAndInstrument(
    inp_path,
    use_vocal_fetch=True,
    use_de_reverb=True,
    use_harmonic_remove=True,
    progress=gr.Progress(),
):
    inp_hash = hashlib.md5(inp_path.encode()).hexdigest()

    jobs = []
    if use_vocal_fetch:
        jobs.append("kim_vocal")
    if use_de_reverb:
        jobs.append("deverb")
    if use_harmonic_remove:
        jobs.append("karaoke")

    vocal_path = f"./tmp/msst_opt/vocal/{inp_hash}_Vocals.wav"
    inst_path = f"./tmp/msst_opt/kim_vocal/{inp_hash}_Instrument.wav"
    real_inst_path = inst_path
    last_vocal = inp_path
    for job in jobs:
        vocal_path = f"./tmp/msst_opt/{job}/{inp_hash}_Vocals.wav"
        inst_path = f"./tmp/msst_opt/{job}/{inp_hash}_Instrument.wav"
        if not os.path.exists(vocal_path) and not os.path.exists(inst_path):
            run_msst(
                inp_path=last_vocal,
                inp_hash=inp_hash,
                vocal_opt_path=vocal_path,
                inst_opt_path=inst_path,
                progress=progress,
                real_type=model_type_to_info[job_to_model_type[job]]["real_type"],
                model_type=job_to_model_type[job],
                progress_desc=I.vocal_separation.job_to_progress_desc[job],
            )
        last_vocal = vocal_path
    print("result", vocal_path, real_inst_path)
    return vocal_path, real_inst_path
